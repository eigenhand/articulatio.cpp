#include <cstdint>
#include "ws_api.h"

#include "breeze/audio.h"
#include "breeze/generation.h"

#include <atomic>
#include <cctype>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <thread>

namespace breeze {

// appends a unicode code point as utf-8
static void utf8_append(std::string & out, uint32_t cp) {
    if (cp < 0x80) {
        out += (char) cp;
    } else if (cp < 0x800) {
        out += (char) (0xC0 | (cp >> 6));
        out += (char) (0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += (char) (0xE0 | (cp >> 12));
        out += (char) (0x80 | ((cp >> 6) & 0x3F));
        out += (char) (0x80 | (cp & 0x3F));
    } else {
        out += (char) (0xF0 | (cp >> 18));
        out += (char) (0x80 | ((cp >> 12) & 0x3F));
        out += (char) (0x80 | ((cp >> 6) & 0x3F));
        out += (char) (0x80 | (cp & 0x3F));
    }
}

// reads four hex digits starting at i, -1 if there are none
static int json_hex4(const std::string & s, size_t i) {
    if (i + 4 > s.size()) return -1;
    int v = 0;
    for (size_t k = i; k < i + 4; k++) {
        const char c = s[k];
        int d;
        if (c >= '0' && c <= '9') d = c - '0';
        else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
        else return -1;
        v = v * 16 + d;
    }
    return v;
}

// the messages are small flat objects, so pulling one field out beats vendoring a json parser
static std::string json_str(const std::string & msg, const char * key) {
    const std::string pat = "\"" + std::string(key) + "\"";
    size_t at = msg.find(pat);
    if (at == std::string::npos) return "";
    at = msg.find(':', at + pat.size());
    if (at == std::string::npos) return "";
    at = msg.find('"', at);
    if (at == std::string::npos) return "";
    std::string out;
    for (size_t i = at + 1; i < msg.size(); i++) {
        const char c = msg[i];
        if (c == '\\' && i + 1 < msg.size()) {
            const char n = msg[++i];
            if (n == 'n') out += '\n';
            else if (n == 't') out += '\t';
            else if (n == 'r') out += '\r';
            else if (n == 'u') {
                // \uXXXX is valid json and used to be silently dropped here, so any client that
                // escapes non-ascii (python's json.dumps does by default) lost every non-ascii
                // character, umlauts included
                const int hi = json_hex4(msg, i + 1);
                if (hi < 0) { continue; }
                i += 4;
                uint32_t cp = (uint32_t) hi;
                // join a surrogate pair (emoji and everything else from U+10000 up)
                if (hi >= 0xD800 && hi <= 0xDBFF && i + 6 < msg.size()
                    && msg[i + 1] == '\\' && msg[i + 2] == 'u') {
                    const int lo = json_hex4(msg, i + 3);
                    if (lo >= 0xDC00 && lo <= 0xDFFF) {
                        cp = 0x10000 + (((uint32_t) hi - 0xD800) << 10) + ((uint32_t) lo - 0xDC00);
                        i += 6;
                    }
                }
                utf8_append(out, cp);
            }
            else out += n;
            continue;
        }
        if (c == '"') break;
        out += c;
    }
    return out;
}

static double json_num(const std::string & msg, const char * key, double def) {
    const std::string pat = "\"" + std::string(key) + "\"";
    size_t at = msg.find(pat);
    if (at == std::string::npos) return def;
    at = msg.find(':', at + pat.size());
    if (at == std::string::npos) return def;
    try {
        return std::stod(msg.substr(at + 1));
    } catch (...) {
        return def;
    }
}

static std::string esc(const std::string & s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n') o += "\\n";
        else if ((unsigned char) c < 0x20) continue;
        else o += c;
    }
    return o;
}

static bool sentence_end(const std::string & s, size_t i) {
    const char c = s[i];
    if (c == '.' || c == '!' || c == '?' || c == ';') {
        // a bare dot inside a number or an abbreviation is not the end of anything
        return i + 1 >= s.size() || s[i + 1] == ' ' || s[i + 1] == '\n';
    }
    // the cjk stops carry their own spacing
    static const char * stops[] = { "\xe3\x80\x82", "\xef\xbc\x81", "\xef\xbc\x9f", "\xef\xbc\x9b" };
    for (const char * st : stops)
        if (s.compare(i, 3, st) == 0) return true;
    return false;
}

// letters and digits, utf-8 continuation bytes not counted (an umlaut counts as one)
static int spoken_chars(const std::string & s) {
    int n = 0;
    for (unsigned char c : s)
        if ((c & 0xC0) != 0x80 && (c >= 0x80 || std::isalnum(c))) n++;
    return n;
}

// a sentence shorter than this (in letters) does not become a piece of its own
static const int kMinPiece = 20;

// moves whole sentences out of buf, leaving a trailing partial behind unless force is set.
//
// every sentence becomes a piece of its own. everything up to the last sentence end used to go out
// as ONE piece, and out of several sentences the model likes to drop the last one. short sentences
// ("Sure!", "Exactly.") are not sent alone though, on their own the model tends to end them right
// away with EOS. they wait in the buffer for the next sentence and go out together with it. only
// at the end (force) does a short remainder go out alone
static std::vector<std::string> drain(std::string & buf, int budget, bool force) {
    std::vector<std::string> out;
    auto emit = [&](const std::string & s) {
        for (std::string & p : split_text(s, budget)) {
            while (!p.empty() && (p.front() == ' ' || p.front() == '\n')) p.erase(p.begin());
            if (!p.empty()) out.push_back(p);
        }
    };

    size_t start = 0, taken = 0;
    std::string cur;
    for (size_t i = 0; i < buf.size(); i++) {
        if (!sentence_end(buf, i)) continue;
        const size_t e = i + ((unsigned char) buf[i] < 0x80 ? 1 : 3);
        cur += buf.substr(start, e - start);
        start = e;
        if (spoken_chars(cur) >= kMinPiece) {
            emit(cur);
            cur.clear();
            taken = e;
        }
    }
    if (force) {
        emit(buf.substr(taken));
        buf.clear();
        return out;
    }
    // nothing finished but the buffer is already long enough to speak, so break it on a space
    if ((int) (buf.size() - taken) > budget) {
        const size_t sp = buf.rfind(' ');
        if (sp != std::string::npos && sp >= taken) {
            emit(buf.substr(taken, sp + 1 - taken));
            taken = sp + 1;
        }
    }
    buf.erase(0, taken);
    return out;
}

namespace {

struct Session {
    GenSession gen;
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> queue;
    std::string buffer;
    std::string instruction = "Speak clearly and naturally.";
    int budget = 600;
    bool started = false;
    bool ending = false;
    bool quit = false;
    bool speaking = false;
    std::atomic<bool> cancel{false};
};

} // namespace

static void event(WsConn & c, const std::string & type, const std::string & extra = "") {
    c.send_text("{\"type\":\"" + type + "\"" + (extra.empty() ? "" : "," + extra) + "}");
}

// drains the queue one piece at a time, taking the gpu lock for each so other connections interleave
static void speaker(WsConn & conn, Session & s, std::mutex & gpu) {
    for (;;) {
        std::string piece;
        {
            std::unique_lock<std::mutex> lock(s.mu);
            s.speaking = false;
            s.cv.wait(lock, [&] { return s.quit || !s.queue.empty(); });
            if (s.quit) return;
            piece = s.queue.front();
            s.queue.pop_front();
            s.speaking = true;
        }
        if (s.cancel) { s.cancel = false; continue; }

        std::unique_lock<std::mutex> hold(gpu, std::try_to_lock);
        if (!hold) {
            event(conn, "queued");
            hold.lock();
        }
        if (s.cancel) { s.cancel = false; continue; }

        {
            std::lock_guard<std::mutex> lock(s.mu);
            s.gen.set_instruction(s.instruction);
        }
        event(conn, "speaking", "\"text\":\"" + esc(piece) + "\"");

        int sent = 0;
        const bool ok = s.gen.speak(piece, [&](const float * a, int n) {
            if (s.cancel || !conn.alive()) return false;
            std::vector<uint8_t> pcm = to_pcm16(a, n);
            sent += n;
            return conn.send_binary(pcm.data(), pcm.size());
        });
        hold.unlock();

        if (s.cancel) { event(conn, "cancelled"); s.cancel = false; continue; }
        if (!ok && !conn.alive()) return;

        std::lock_guard<std::mutex> lock(s.mu);
        if (s.queue.empty() && s.ending) {
            s.ending = false;
            event(conn, "done");
        }
    }
}

static void handle_start(WsConn & conn, Session & s, const std::string & msg, BreezeModel & model,
                         MimiCodec & codec, VoiceStore & store, int chunk_first, int chunk_max,
                         int split_chars,
                         float def_speed, int def_head, int def_tail, int def_pause) {
    GenRequest g;
    g.instruction = json_str(msg, "instruction");
    if (g.instruction.empty()) g.instruction = "Speak clearly and naturally.";
    g.ref_text = json_str(msg, "ref_text");
    g.cfg_scale = (float) json_num(msg, "cfg_scale", 1.0);
    g.seed = (int) json_num(msg, "seed", 42);
    g.temperature = (float) json_num(msg, "temperature", 0);
    g.top_k = (int) json_num(msg, "top_k", 0);
    g.chunk_first = chunk_first;
    g.chunk_max = chunk_max;

    const std::string vid = json_str(msg, "voice_id");
    if (!vid.empty() && !store.take(vid, g.ref_codes, g.ref_frames, g.ref_text)) {
        event(conn, "error", "\"message\":\"unknown voice_id\"");
        return;
    }

    g.rolling_anchor = json_num(msg, "rolling_anchor", 0) != 0;
    g.carry_cache = json_num(msg, "carry_cache", 0) != 0;
    g.carry_eos = json_num(msg, "carry_eos", 1) != 0;
    g.cache_seq = (int) json_num(msg, "cache_seq", 2048);
    g.speed = (float) json_num(msg, "speed", def_speed);
    g.trim_head_ms = (int) json_num(msg, "trim_head_ms", def_head);
    g.trim_tail_ms = (int) json_num(msg, "trim_tail_ms", def_tail);
    g.pause_ms = (int) json_num(msg, "pause_ms", def_pause);

    std::lock_guard<std::mutex> lock(s.mu);
    s.instruction = g.instruction;
    s.budget = (int) json_num(msg, "split_chars", split_chars);
    // streaming drains sentence by sentence, so it always needs a real budget to aim at
    if (s.budget <= 0) s.budget = 600;
    s.gen.begin(model, codec, g);
    s.started = true;
    s.queue.clear();
    s.buffer.clear();
    s.cancel = false;
    event(conn, "started", "\"voice_id\":\"" + esc(vid) + "\"");
}

void ws_connection(WsConn & conn, BreezeModel & model, MimiCodec & codec, VoiceStore & store,
                   std::mutex & gpu, int chunk_first, int chunk_max, int split_chars,
                   float def_speed, int def_head, int def_tail, int def_pause) {
    Session s;
    std::thread worker([&] { speaker(conn, s, gpu); });
    event(conn, "ready", "\"sample_rate\":24000,\"format\":\"s16le\"");

    std::string msg;
    bool binary = false;
    while (conn.recv(msg, binary)) {
        if (binary) continue;
        const std::string type = json_str(msg, "type");

        if (type == "start") {
            handle_start(conn, s, msg, model, codec, store, chunk_first, chunk_max, split_chars,
                         def_speed, def_head, def_tail, def_pause);
            continue;
        }
        if (!s.started) {
            event(conn, "error", "\"message\":\"send start first\"");
            continue;
        }
        if (type == "instruction") {
            std::lock_guard<std::mutex> lock(s.mu);
            // takes effect on the next piece, whatever is already being spoken finishes as it was
            s.instruction = json_str(msg, "instruction");
            event(conn, "instruction_set");
        } else if (type == "text" || type == "flush" || type == "end") {
            std::unique_lock<std::mutex> lock(s.mu);
            s.buffer += json_str(msg, "text");
            const bool force = type != "text";
            // while there is no clip to clone the opening piece doubles as the reference, and a
            // long one makes the model skip sentences later, so it stays near a normal clip length
            const int budget = s.gen.needs_anchor() && s.queue.empty() ? 200 : s.budget;
            for (std::string & p : drain(s.buffer, budget, force)) s.queue.push_back(p);
            if (type == "end") s.ending = true;
            lock.unlock();
            s.cv.notify_one();
        } else if (type == "cancel") {
            std::lock_guard<std::mutex> lock(s.mu);
            s.queue.clear();
            s.buffer.clear();
            s.ending = false;
            // only latch it while something is actually being spoken, otherwise the flag
            // sits true and swallows whatever gets sent next
            s.cancel = s.speaking;
        } else {
            event(conn, "error", "\"message\":\"unknown type\"");
        }
    }

    {
        std::lock_guard<std::mutex> lock(s.mu);
        s.quit = true;
        s.cancel = true;
    }
    s.cv.notify_all();
    worker.join();
}

} // namespace breeze
