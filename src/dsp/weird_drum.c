/* ============================================================================
 * Weird Drum Machine — 8-voice analog drum synthesizer for Ableton Move
 * Port of dfilaretti/WeirdDrums (MIT) to Move Everything framework
 *
 * Architecture: 8 independent drum voices, each with:
 *   - Phase-accumulator oscillator (sine/saw/square)
 *   - Exponential AD envelope (amp)
 *   - Pitch envelope + pitch LFO
 *   - White noise generator with SVF filter (LP/HP/BP)
 *   - Noise AD envelope
 *   - tanh distortion
 *   - Per-voice level
 *
 * Master bus: compressor, distortion, 3-band EQ
 * UI: Mixer page, General page, 8 voice pages
 * ============================================================================ */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define SAMPLE_RATE     44100.0f
#define BLOCK_SIZE      128
#define NUM_VOICES      8
#define TWO_PI          6.283185307f
#define ENV_MIN         0.0001f

/* ── Clamp ── */
static inline float clampf(float x, float lo, float hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

/* ── One-pole smoother ── */
static inline float onepole(float *state, float target, float coeff) {
    *state += coeff * (target - *state);
    return *state;
}

/* ============================================================================
 * DSP Primitives
 * ============================================================================ */

/* ── Exponential AD Envelope ── */
typedef struct {
    enum { ENV_IDLE, ENV_ATTACK, ENV_DECAY, ENV_TAILOFF } state;
    float value;
    float attack_mult;
    float decay_mult;
    int   attack_len;
    int   decay_len;
    int   sample_idx;
    float tailoff;
} wd_envelope_t;

static void env_set_params(wd_envelope_t *e, float attack_sec, float decay_sec, float sr) {
    e->attack_len = (int)(attack_sec * sr);
    e->decay_len  = (int)(decay_sec * sr);
    if (e->attack_len < 1) e->attack_len = 1;
    if (e->decay_len < 1)  e->decay_len = 1;
    e->attack_mult = 1.0f + (logf(1.0f) - logf(ENV_MIN)) / (float)e->attack_len;
    e->decay_mult  = 1.0f + (logf(ENV_MIN) - logf(1.0f)) / (float)e->decay_len;
}

static void env_reset(wd_envelope_t *e) {
    e->state = ENV_IDLE;
    e->value = ENV_MIN;
    e->sample_idx = 0;
    e->tailoff = 0.0f;
}

static void env_note_on(wd_envelope_t *e) {
    e->value = ENV_MIN;
    e->sample_idx = 0;
    e->state = ENV_ATTACK;
}

static void env_note_off(wd_envelope_t *e) {
    if (e->state != ENV_IDLE) {
        e->tailoff = 1.0f;
        e->state = ENV_TAILOFF;
    }
}

static float env_next(wd_envelope_t *e) {
    if (e->state == ENV_IDLE) return 0.0f;

    if (e->state == ENV_ATTACK) {
        e->value *= e->attack_mult;
        e->sample_idx++;
        if (e->sample_idx >= e->attack_len) {
            e->value = 1.0f;
            e->sample_idx = 0;
            e->state = ENV_DECAY;
        }
    } else if (e->state == ENV_DECAY) {
        e->value *= e->decay_mult;
        e->sample_idx++;
        if (e->sample_idx >= e->decay_len) {
            env_reset(e);
        }
    } else if (e->state == ENV_TAILOFF) {
        e->value *= e->tailoff;
        e->tailoff *= 0.99f;
        if (e->tailoff <= 0.005f) env_reset(e);
    }

    return e->value;
}

static int env_active(const wd_envelope_t *e) {
    return e->state != ENV_IDLE;
}

/* ── Phase Accumulator Oscillator with continuous waveform morphing ──
 * wave 0.0 = sine, 0.33 = triangle, 0.66 = sawtooth, 1.0 = square
 * Crossfades smoothly between adjacent waveforms */
typedef struct {
    float phase;      /* 0..1 */
} wd_osc_t;

static void osc_reset(wd_osc_t *o) { o->phase = 0.0f; }

static float osc_next(wd_osc_t *o, float freq, float wave) {
    float p = o->phase;

    /* Generate all 4 waveforms from phase */
    float sine = sinf(p * TWO_PI);
    float tri  = (p < 0.5f) ? (4.0f * p - 1.0f) : (3.0f - 4.0f * p);
    float saw  = 2.0f * p - 1.0f;
    float sqr  = (p < 0.5f) ? 1.0f : -1.0f;

    /* Morph: crossfade between adjacent waveforms */
    float out;
    float w = clampf(wave, 0.0f, 1.0f) * 3.0f;  /* 0..3 */
    if (w < 1.0f) {
        out = sine * (1.0f - w) + tri * w;        /* sine → triangle */
    } else if (w < 2.0f) {
        float t = w - 1.0f;
        out = tri * (1.0f - t) + saw * t;          /* triangle → sawtooth */
    } else {
        float t = w - 2.0f;
        out = saw * (1.0f - t) + sqr * t;          /* sawtooth → square */
    }

    o->phase += freq / SAMPLE_RATE;
    if (o->phase >= 1.0f) o->phase -= 1.0f;
    return out;
}

/* ── White Noise (xorshift32) ── */
typedef struct { uint32_t state; } wd_noise_t;

static float noise_next(wd_noise_t *n) {
    uint32_t x = n->state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    n->state = x;
    return ((float)(x & 0xFFFF) / 32768.0f) - 1.0f;
}

/* ── State Variable Filter (LP/HP/BP) ── */
typedef struct {
    float lp, bp, hp;
    int   type; /* 0=LP, 1=HP, 2=BP */
} wd_svf_t;

static void svf_reset(wd_svf_t *f) { f->lp = f->bp = f->hp = 0.0f; }

static float svf_process(wd_svf_t *f, float in, float cutoff, float res) {
    float fc = 2.0f * sinf(3.14159265f * cutoff / SAMPLE_RATE);
    fc = clampf(fc, 0.0f, 1.0f);
    float q = 1.0f / clampf(res, 0.5f, 20.0f);
    f->lp += fc * f->bp;
    f->hp  = in - f->lp - q * f->bp;
    f->bp += fc * f->hp;
    switch (f->type) {
        case 1:  return f->hp;
        case 2:  return f->bp;
        default: return f->lp;
    }
}

/* ── Pitch LFO (sine) ── */
typedef struct {
    float phase;
} wd_lfo_t;

static float lfo_next(wd_lfo_t *l, float rate) {
    float out = sinf(l->phase * TWO_PI);
    l->phase += rate / SAMPLE_RATE;
    if (l->phase >= 1.0f) l->phase -= 1.0f;
    return out;
}

/* ============================================================================
 * Drum Voice
 * ============================================================================ */

typedef struct {
    /* DSP modules */
    wd_osc_t      osc;
    wd_envelope_t amp_env;
    wd_envelope_t noise_env;
    wd_envelope_t pitch_env;
    wd_noise_t    noise;
    wd_svf_t      filter;
    wd_lfo_t      lfo;

    /* Parameters */
    float freq;             /* 20..20000 Hz */
    float attack;           /* 0.0001..1.0 sec */
    float decay;            /* 0.0001..4.0 sec */
    float wave;             /* 0=sine, 0.33=tri, 0.66=saw, 1.0=square */
    float pitch_env_amt;    /* 0..1 (scales 0..1000 Hz) */
    float pitch_env_rate;   /* 0.001..2.0 sec */
    float pitch_lfo_amt;    /* 0..1 */
    float pitch_lfo_rate;   /* 0.1..80 Hz */
    int   filter_type;      /* 0=LP, 1=HP, 2=BP */
    float filter_cutoff;    /* 20..18000 Hz */
    float filter_res;       /* 1..5 */
    float noise_attack;     /* 0.0001..1.0 sec */
    float noise_decay;      /* 0.0001..1.0 sec */
    float mix;              /* 0..1 (0=osc, 1=noise) */
    float distortion;       /* 0..50 dB */
    float level;            /* 0..1 linear */
    float pan;              /* -1..+1 (left..right, 0=center) */

    int   preset;
    int   active;           /* voice is sounding */
    float velocity;         /* last note-on velocity */

    /* Clap retrigger mechanism (808/909 style flutter) */
    int   clap_count;       /* remaining retriggers (0=off, 1-4=flutter hits) */
    int   clap_timer;       /* samples until next retrigger */
    int   clap_spacing;     /* samples between retriggers (~10ms = 441 samples) */
} wd_voice_t;

/* ── 41 Preset shapes: 5×8 categories + Custom ── */
#define NUM_PRESETS 41

/* Helper: set common defaults, then preset overrides */
static void preset_defaults(wd_voice_t *v) {
    v->attack = 0.001f; v->pitch_lfo_amt = 0.0f; v->pitch_lfo_rate = 1.0f;
    v->noise_attack = 0.001f; v->clap_count = 0; v->clap_spacing = 0;
}

static void voice_apply_preset(wd_voice_t *v, int preset) {
    v->preset = preset;
    preset_defaults(v);
    switch (preset) {
    /* ── KICKS (0-4) ── */
    case 0: /* Sub Kick — deep 808 sub */
        v->freq=38; v->wave=0.0f; v->decay=0.8f; v->pitch_env_amt=0.95f; v->pitch_env_rate=0.07f;
        v->filter_type=0; v->filter_cutoff=250; v->filter_res=1.2f;
        v->noise_decay=0.02f; v->mix=0.02f; v->distortion=5; v->level=0.95f; break;
    case 1: /* Punch Kick — tight 909 */
        v->freq=65; v->wave=0.0f; v->attack=0.0003f; v->decay=0.15f; v->pitch_env_amt=0.75f; v->pitch_env_rate=0.02f;
        v->filter_type=0; v->filter_cutoff=600; v->filter_res=1.5f;
        v->noise_decay=0.015f; v->mix=0.1f; v->distortion=14; v->level=0.9f; break;
    case 2: /* FM Kick — saw sweep, Microtonic */
        v->freq=48; v->wave=0.66f; v->attack=0.0001f; v->decay=0.35f; v->pitch_env_amt=1.0f; v->pitch_env_rate=0.04f;
        v->pitch_lfo_amt=0.03f; v->pitch_lfo_rate=8;
        v->filter_type=0; v->filter_cutoff=800; v->filter_res=2.0f;
        v->noise_decay=0.04f; v->mix=0.08f; v->distortion=20; v->level=0.85f; break;
    case 3: /* Dusty Kick — lo-fi saturated */
        v->freq=50; v->wave=0.15f; v->decay=0.2f; v->pitch_env_amt=0.6f; v->pitch_env_rate=0.025f;
        v->filter_type=0; v->filter_cutoff=500; v->filter_res=1.3f;
        v->noise_decay=0.06f; v->mix=0.12f; v->distortion=25; v->level=0.85f; break;
    case 4: /* Long Kick — boomy tail */
        v->freq=42; v->wave=0.0f; v->decay=1.2f; v->pitch_env_amt=0.85f; v->pitch_env_rate=0.08f;
        v->filter_type=0; v->filter_cutoff=350; v->filter_res=1.0f;
        v->noise_decay=0.03f; v->mix=0.03f; v->distortion=3; v->level=0.9f; break;

    /* ── SNARES (5-9) ── */
    case 5: /* Snare — classic analog */
        v->freq=190; v->wave=0.0f; v->attack=0.0005f; v->decay=0.16f; v->pitch_env_amt=0.35f; v->pitch_env_rate=0.025f;
        v->filter_type=2; v->filter_cutoff=3500; v->filter_res=1.4f;
        v->noise_decay=0.16f; v->mix=0.6f; v->distortion=4; v->level=0.8f; break;
    case 6: /* Snap Snare — crispy 909 */
        v->freq=220; v->wave=0.0f; v->attack=0.0003f; v->decay=0.1f; v->pitch_env_amt=0.4f; v->pitch_env_rate=0.015f;
        v->filter_type=1; v->filter_cutoff=4000; v->filter_res=1.8f;
        v->noise_attack=0.0003f; v->noise_decay=0.12f; v->mix=0.7f; v->distortion=6; v->level=0.75f; break;
    case 7: /* Fat Snare — thick body */
        v->freq=150; v->wave=0.15f; v->decay=0.25f; v->pitch_env_amt=0.2f; v->pitch_env_rate=0.04f;
        v->filter_type=0; v->filter_cutoff=2000; v->filter_res=1.0f;
        v->noise_decay=0.2f; v->mix=0.45f; v->distortion=5; v->level=0.85f; break;
    case 8: /* Crack Snare — harsh transient */
        v->freq=350; v->wave=1.0f; v->attack=0.0001f; v->decay=0.06f; v->pitch_env_amt=0.5f; v->pitch_env_rate=0.008f;
        v->filter_type=2; v->filter_cutoff=5000; v->filter_res=3.5f;
        v->noise_attack=0.0001f; v->noise_decay=0.08f; v->mix=0.75f; v->distortion=16; v->level=0.7f; break;
    case 9: /* Noise Snare — pure noise */
        v->freq=300; v->wave=1.0f; v->attack=0.0001f; v->decay=0.12f; v->pitch_env_amt=0.0f; v->pitch_env_rate=0.01f;
        v->filter_type=2; v->filter_cutoff=4500; v->filter_res=1.2f;
        v->noise_attack=0.0001f; v->noise_decay=0.15f; v->mix=0.95f; v->distortion=2; v->level=0.7f; break;

    /* ── TOMS (10-14) ── */
    case 10: /* Low Tom */
        v->freq=75; v->wave=0.0f; v->decay=0.4f; v->pitch_env_amt=0.55f; v->pitch_env_rate=0.045f;
        v->filter_type=0; v->filter_cutoff=700; v->filter_res=1.0f;
        v->noise_decay=0.08f; v->mix=0.1f; v->distortion=3; v->level=0.8f; break;
    case 11: /* Mid Tom */
        v->freq=130; v->wave=0.0f; v->decay=0.3f; v->pitch_env_amt=0.45f; v->pitch_env_rate=0.035f;
        v->filter_type=0; v->filter_cutoff=900; v->filter_res=1.0f;
        v->noise_decay=0.06f; v->mix=0.08f; v->distortion=2; v->level=0.8f; break;
    case 12: /* High Tom */
        v->freq=200; v->wave=0.0f; v->decay=0.22f; v->pitch_env_amt=0.4f; v->pitch_env_rate=0.03f;
        v->filter_type=0; v->filter_cutoff=1200; v->filter_res=1.0f;
        v->noise_decay=0.05f; v->mix=0.08f; v->distortion=2; v->level=0.75f; break;
    case 13: /* Acid Tom — resonant filter sweep */
        v->freq=110; v->wave=0.66f; v->decay=0.35f; v->pitch_env_amt=0.7f; v->pitch_env_rate=0.06f;
        v->filter_type=0; v->filter_cutoff=1800; v->filter_res=3.8f;
        v->noise_decay=0.05f; v->mix=0.05f; v->distortion=10; v->level=0.75f; break;
    case 14: /* Conga — long resonant body */
        v->freq=240; v->wave=0.0f; v->attack=0.0005f; v->decay=0.5f; v->pitch_env_amt=0.25f; v->pitch_env_rate=0.02f;
        v->filter_type=2; v->filter_cutoff=1200; v->filter_res=2.5f;
        v->noise_attack=0.0005f; v->noise_decay=0.06f; v->mix=0.15f; v->distortion=6; v->level=0.7f; break;

    /* ── HI-HATS (15-19) ── */
    case 15: /* Closed HH — tight */
        v->freq=450; v->wave=1.0f; v->attack=0.0001f; v->decay=0.035f; v->pitch_env_amt=0.0f; v->pitch_env_rate=0.01f;
        v->pitch_lfo_amt=0.04f; v->pitch_lfo_rate=60;
        v->filter_type=1; v->filter_cutoff=9000; v->filter_res=2.0f;
        v->noise_attack=0.0001f; v->noise_decay=0.035f; v->mix=0.92f; v->distortion=3; v->level=0.55f; break;
    case 16: /* Open HH — sizzle */
        v->freq=420; v->wave=1.0f; v->attack=0.0001f; v->decay=0.35f; v->pitch_env_amt=0.0f; v->pitch_env_rate=0.01f;
        v->pitch_lfo_amt=0.06f; v->pitch_lfo_rate=40;
        v->filter_type=1; v->filter_cutoff=6500; v->filter_res=1.8f;
        v->noise_attack=0.0001f; v->noise_decay=0.35f; v->mix=0.93f; v->distortion=2; v->level=0.5f; break;
    case 17: /* Pedal HH — medium */
        v->freq=380; v->wave=1.0f; v->attack=0.0001f; v->decay=0.1f; v->pitch_env_amt=0.0f; v->pitch_env_rate=0.01f;
        v->pitch_lfo_amt=0.03f; v->pitch_lfo_rate=50;
        v->filter_type=1; v->filter_cutoff=7000; v->filter_res=1.5f;
        v->noise_attack=0.0001f; v->noise_decay=0.1f; v->mix=0.9f; v->distortion=1; v->level=0.5f; break;
    case 18: /* Metallic HH — LXR-02 */
        v->freq=500; v->wave=1.0f; v->attack=0.0001f; v->decay=0.05f; v->pitch_env_amt=0.05f; v->pitch_env_rate=0.005f;
        v->pitch_lfo_amt=0.1f; v->pitch_lfo_rate=75;
        v->filter_type=2; v->filter_cutoff=8000; v->filter_res=3.0f;
        v->noise_attack=0.0001f; v->noise_decay=0.05f; v->mix=0.85f; v->distortion=8; v->level=0.55f; break;
    case 19: /* Trashy HH — dirty, distorted */
        v->freq=550; v->wave=0.8f; v->attack=0.0001f; v->decay=0.08f; v->pitch_env_amt=0.0f; v->pitch_env_rate=0.01f;
        v->pitch_lfo_amt=0.15f; v->pitch_lfo_rate=65;
        v->filter_type=1; v->filter_cutoff=5000; v->filter_res=2.5f;
        v->noise_attack=0.0001f; v->noise_decay=0.08f; v->mix=0.88f; v->distortion=18; v->level=0.5f; break;

    /* ── CYMBALS (20-24) ── */
    case 20: /* Crash — bright */
        v->freq=280; v->wave=1.0f; v->attack=0.002f; v->decay=1.2f; v->pitch_env_amt=0.0f; v->pitch_env_rate=0.01f;
        v->pitch_lfo_amt=0.07f; v->pitch_lfo_rate=4;
        v->filter_type=1; v->filter_cutoff=4500; v->filter_res=1.2f;
        v->noise_attack=0.002f; v->noise_decay=1.2f; v->mix=0.88f; v->distortion=0; v->level=0.45f; break;
    case 21: /* Ride — bell character */
        v->freq=600; v->wave=1.0f; v->attack=0.0001f; v->decay=0.8f; v->pitch_env_amt=0.04f; v->pitch_env_rate=0.005f;
        v->pitch_lfo_amt=0.03f; v->pitch_lfo_rate=6;
        v->filter_type=2; v->filter_cutoff=5500; v->filter_res=2.0f;
        v->noise_attack=0.0001f; v->noise_decay=0.6f; v->mix=0.7f; v->distortion=4; v->level=0.5f; break;
    case 22: /* Ride Bell — ping */
        v->freq=900; v->wave=1.0f; v->attack=0.0001f; v->decay=0.5f; v->pitch_env_amt=0.08f; v->pitch_env_rate=0.004f;
        v->filter_type=2; v->filter_cutoff=4000; v->filter_res=3.5f;
        v->noise_attack=0.0001f; v->noise_decay=0.2f; v->mix=0.4f; v->distortion=8; v->level=0.55f; break;
    case 23: /* Dark Crash — filtered */
        v->freq=250; v->wave=0.85f; v->attack=0.003f; v->decay=1.5f; v->pitch_env_amt=0.0f; v->pitch_env_rate=0.01f;
        v->pitch_lfo_amt=0.05f; v->pitch_lfo_rate=3;
        v->filter_type=0; v->filter_cutoff=3000; v->filter_res=1.0f;
        v->noise_attack=0.003f; v->noise_decay=1.5f; v->mix=0.85f; v->distortion=0; v->level=0.45f; break;
    case 24: /* Sizzle — long shimmer */
        v->freq=350; v->wave=1.0f; v->attack=0.005f; v->decay=2.5f; v->pitch_env_amt=0.0f; v->pitch_env_rate=0.01f;
        v->pitch_lfo_amt=0.1f; v->pitch_lfo_rate=5;
        v->filter_type=1; v->filter_cutoff=6000; v->filter_res=1.5f;
        v->noise_attack=0.005f; v->noise_decay=2.5f; v->mix=0.9f; v->distortion=0; v->level=0.4f; break;

    /* ── CLAPS (25-29) — use clap retrigger for flutter ── */
    case 25: /* Clap 808 — classic triple-hit flutter */
        v->freq=800; v->wave=1.0f; v->attack=0.0001f; v->decay=0.2f; v->pitch_env_amt=0.0f; v->pitch_env_rate=0.01f;
        v->filter_type=2; v->filter_cutoff=1800; v->filter_res=1.5f;
        v->noise_attack=0.0001f; v->noise_decay=0.2f; v->mix=0.95f; v->distortion=4; v->level=0.7f;
        v->clap_count=3; v->clap_spacing=480; break; /* 3 hits, ~11ms apart */
    case 26: /* Tight Clap — fast flutter, short tail */
        v->freq=1000; v->wave=1.0f; v->attack=0.0001f; v->decay=0.1f; v->pitch_env_amt=0.0f; v->pitch_env_rate=0.01f;
        v->filter_type=2; v->filter_cutoff=2500; v->filter_res=1.8f;
        v->noise_attack=0.0001f; v->noise_decay=0.1f; v->mix=0.92f; v->distortion=6; v->level=0.7f;
        v->clap_count=2; v->clap_spacing=350; break;
    case 27: /* Big Clap — wide flutter, long reverby tail */
        v->freq=700; v->wave=1.0f; v->attack=0.0001f; v->decay=0.4f; v->pitch_env_amt=0.0f; v->pitch_env_rate=0.01f;
        v->filter_type=2; v->filter_cutoff=1500; v->filter_res=1.2f;
        v->noise_attack=0.0001f; v->noise_decay=0.4f; v->mix=0.93f; v->distortion=3; v->level=0.65f;
        v->clap_count=4; v->clap_spacing=550; break; /* 4 hits, wider spacing */
    case 28: /* Dirty Clap — distorted */
        v->freq=900; v->wave=0.85f; v->attack=0.0001f; v->decay=0.15f; v->pitch_env_amt=0.0f; v->pitch_env_rate=0.01f;
        v->filter_type=2; v->filter_cutoff=2000; v->filter_res=2.0f;
        v->noise_attack=0.0001f; v->noise_decay=0.15f; v->mix=0.9f; v->distortion=18; v->level=0.65f;
        v->clap_count=3; v->clap_spacing=400; break;
    case 29: /* Snap Clap — single hit, no flutter */
        v->freq=1200; v->wave=1.0f; v->attack=0.0001f; v->decay=0.08f; v->pitch_env_amt=0.1f; v->pitch_env_rate=0.005f;
        v->filter_type=1; v->filter_cutoff=3000; v->filter_res=2.5f;
        v->noise_attack=0.0001f; v->noise_decay=0.08f; v->mix=0.85f; v->distortion=8; v->level=0.7f; break;

    /* ── PERCUSSION (30-34) ── */
    case 30: /* Rimshot — sharp click */
        v->freq=500; v->wave=1.0f; v->attack=0.0001f; v->decay=0.035f; v->pitch_env_amt=0.25f; v->pitch_env_rate=0.008f;
        v->filter_type=2; v->filter_cutoff=4000; v->filter_res=2.5f;
        v->noise_attack=0.0001f; v->noise_decay=0.03f; v->mix=0.4f; v->distortion=8; v->level=0.75f; break;
    case 31: /* Cowbell — metallic ring */
        v->freq=560; v->wave=1.0f; v->attack=0.0001f; v->decay=0.12f; v->pitch_env_amt=0.06f; v->pitch_env_rate=0.004f;
        v->filter_type=2; v->filter_cutoff=2200; v->filter_res=3.5f;
        v->noise_attack=0.0001f; v->noise_decay=0.03f; v->mix=0.12f; v->distortion=14; v->level=0.65f; break;
    case 32: /* Clave — woody click */
        v->freq=2500; v->wave=0.0f; v->attack=0.0001f; v->decay=0.02f; v->pitch_env_amt=0.15f; v->pitch_env_rate=0.005f;
        v->filter_type=2; v->filter_cutoff=3000; v->filter_res=3.0f;
        v->noise_attack=0.0001f; v->noise_decay=0.01f; v->mix=0.1f; v->distortion=8; v->level=0.7f; break;
    case 33: /* Shaker — maracas */
        v->freq=600; v->wave=1.0f; v->attack=0.002f; v->decay=0.12f; v->pitch_env_amt=0.0f; v->pitch_env_rate=0.01f;
        v->filter_type=1; v->filter_cutoff=5000; v->filter_res=1.0f;
        v->noise_attack=0.002f; v->noise_decay=0.12f; v->mix=0.85f; v->distortion=0; v->level=0.5f; break;
    case 34: /* Tamb — tambourine */
        v->freq=700; v->wave=1.0f; v->attack=0.0001f; v->decay=0.2f; v->pitch_env_amt=0.0f; v->pitch_env_rate=0.01f;
        v->pitch_lfo_amt=0.08f; v->pitch_lfo_rate=50;
        v->filter_type=1; v->filter_cutoff=7000; v->filter_res=2.0f;
        v->noise_attack=0.0001f; v->noise_decay=0.2f; v->mix=0.9f; v->distortion=2; v->level=0.5f; break;

    /* ── FX (35-39) ── */
    case 35: /* Zap — laser sweep */
        v->freq=1200; v->wave=0.66f; v->attack=0.0001f; v->decay=0.1f; v->pitch_env_amt=1.0f; v->pitch_env_rate=0.1f;
        v->filter_type=0; v->filter_cutoff=8000; v->filter_res=2.5f;
        v->noise_attack=0.0001f; v->noise_decay=0.01f; v->mix=0.03f; v->distortion=18; v->level=0.7f; break;
    case 36: /* Glitch — fast LFO chaos */
        v->freq=320; v->wave=1.0f; v->attack=0.0001f; v->decay=0.07f; v->pitch_env_amt=0.4f; v->pitch_env_rate=0.015f;
        v->pitch_lfo_amt=0.3f; v->pitch_lfo_rate=70;
        v->filter_type=2; v->filter_cutoff=3000; v->filter_res=3.0f;
        v->noise_attack=0.0001f; v->noise_decay=0.07f; v->mix=0.5f; v->distortion=22; v->level=0.6f; break;
    case 37: /* Drone Hit — long throb */
        v->freq=60; v->wave=0.66f; v->attack=0.01f; v->decay=2.0f; v->pitch_env_amt=0.15f; v->pitch_env_rate=0.15f;
        v->pitch_lfo_amt=0.12f; v->pitch_lfo_rate=3.5f;
        v->filter_type=0; v->filter_cutoff=1500; v->filter_res=3.0f;
        v->noise_attack=0.01f; v->noise_decay=0.5f; v->mix=0.25f; v->distortion=8; v->level=0.65f; break;
    case 38: /* Blip — short tonal ping */
        v->freq=1800; v->wave=0.0f; v->attack=0.0001f; v->decay=0.025f; v->pitch_env_amt=0.6f; v->pitch_env_rate=0.008f;
        v->filter_type=2; v->filter_cutoff=5000; v->filter_res=3.5f;
        v->noise_attack=0.0001f; v->noise_decay=0.01f; v->mix=0.05f; v->distortion=10; v->level=0.6f; break;
    case 39: /* Noise Burst — white noise hit */
        v->freq=200; v->wave=1.0f; v->attack=0.0001f; v->decay=0.06f; v->pitch_env_amt=0.0f; v->pitch_env_rate=0.01f;
        v->filter_type=2; v->filter_cutoff=4000; v->filter_res=1.0f;
        v->noise_attack=0.0001f; v->noise_decay=0.06f; v->mix=1.0f; v->distortion=2; v->level=0.6f; break;

    default: /* 40 = Custom — no change */ break;
    }
}

static void voice_init(wd_voice_t *v, int idx) {
    memset(v, 0, sizeof(*v));
    v->noise.state = 123456789u + (uint32_t)idx * 987654u;
    v->level = 0.8f;
    /* Default kit: Kick, Snare, Low Tom, Clap 808, Rimshot, Closed HH, Open HH, Crash */
    int default_presets[] = { 0, 5, 10, 25, 30, 15, 16, 20 };
    voice_apply_preset(v, default_presets[idx]);
}

static void voice_trigger(wd_voice_t *v, float velocity) {
    v->velocity = velocity;
    v->active = 1;

    /* Reset oscillator phase */
    osc_reset(&v->osc);

    /* Configure envelopes */
    env_set_params(&v->amp_env, v->attack, v->decay, SAMPLE_RATE);
    env_set_params(&v->noise_env, v->noise_attack, v->noise_decay, SAMPLE_RATE);
    env_set_params(&v->pitch_env, 0.001f, v->pitch_env_rate, SAMPLE_RATE);

    env_note_on(&v->amp_env);
    env_note_on(&v->noise_env);
    env_note_on(&v->pitch_env);

    /* Setup filter */
    v->filter.type = v->filter_type;

    /* Clap retrigger: schedule flutter hits */
    if (v->clap_count > 0) {
        v->clap_timer = v->clap_spacing;
    }
}

static float voice_render_sample(wd_voice_t *v) {
    if (!v->active) return 0.0f;

    /* Clap retrigger: re-fire noise envelope at scheduled intervals */
    if (v->clap_count > 0) {
        if (--v->clap_timer <= 0) {
            v->clap_count--;
            v->clap_timer = v->clap_spacing;
            /* Re-trigger noise envelope for the flutter hit */
            env_set_params(&v->noise_env, 0.0001f, v->noise_decay, SAMPLE_RATE);
            env_note_on(&v->noise_env);
        }
    }

    /* Check if voice is done */
    if (!env_active(&v->amp_env) && !env_active(&v->noise_env) && v->clap_count <= 0) {
        v->active = 0;
        return 0.0f;
    }

    /* Pitch modulation: LFO + pitch envelope */
    float freq = v->freq;

    /* Pitch LFO: freq * 2^(lfo * amount) */
    if (v->pitch_lfo_amt > 0.001f) {
        float lfo_val = lfo_next(&v->lfo, v->pitch_lfo_rate);
        freq *= powf(2.0f, lfo_val * v->pitch_lfo_amt);
    }

    /* Pitch envelope: sweep from freq to freq + 1000*amount */
    if (v->pitch_env_amt > 0.001f) {
        float pe = env_next(&v->pitch_env);
        float fmax = clampf(freq + 1000.0f * v->pitch_env_amt, 0.0f, 20000.0f);
        freq = freq + pe * (fmax - freq);
    } else {
        /* Still need to advance pitch env */
        env_next(&v->pitch_env);
    }

    freq = clampf(freq, 20.0f, 20000.0f);

    /* Oscillator path: osc * amp_env * velocity * (1-mix) */
    float osc_out = osc_next(&v->osc, freq, v->wave);
    float amp = env_next(&v->amp_env) * v->velocity;
    float osc_signal = osc_out * amp * (1.0f - v->mix);

    /* Noise path: noise -> SVF -> noise_env * velocity * mix */
    float n = noise_next(&v->noise);
    float filtered = svf_process(&v->filter, n, v->filter_cutoff, v->filter_res);
    float noise_amp = env_next(&v->noise_env) * v->velocity;
    float noise_signal = filtered * noise_amp * v->mix;

    /* Mix */
    float out = osc_signal + noise_signal;

    /* Distortion: pregain -> tanh */
    if (v->distortion > 0.1f) {
        float pregain = powf(10.0f, v->distortion / 20.0f);
        out = tanhf(out * pregain);
    }

    return out * v->level;
}

/* ============================================================================
 * Master Bus FX — Massenburg 8200-style parametric EQ + compressor + distortion
 * ============================================================================ */

/* ── Biquad filter (peaking EQ, Massenburg 8200 constant-Q bell) ── */
typedef struct {
    float b0, b1, b2, a1, a2;  /* coefficients (normalized) */
    float x1, x2, y1, y2;      /* delay states */
} wd_biquad_t;

static void biquad_reset(wd_biquad_t *bq) {
    bq->x1 = bq->x2 = bq->y1 = bq->y2 = 0.0f;
}

/* Peaking EQ coefficients — Audio EQ Cookbook (Robert Bristow-Johnson) */
static void biquad_set_peaking(wd_biquad_t *bq, float freq, float gain_db, float Q) {
    if (Q < 0.1f) Q = 0.1f;
    float A = powf(10.0f, gain_db / 40.0f);  /* sqrt of linear gain */
    float w0 = TWO_PI * freq / SAMPLE_RATE;
    float sinw = sinf(w0);
    float cosw = cosf(w0);
    float alpha = sinw / (2.0f * Q);

    float b0 = 1.0f + alpha * A;
    float b1 = -2.0f * cosw;
    float b2 = 1.0f - alpha * A;
    float a0 = 1.0f + alpha / A;
    float a1 = -2.0f * cosw;
    float a2 = 1.0f - alpha / A;

    /* Normalize */
    float inv_a0 = 1.0f / a0;
    bq->b0 = b0 * inv_a0;
    bq->b1 = b1 * inv_a0;
    bq->b2 = b2 * inv_a0;
    bq->a1 = a1 * inv_a0;
    bq->a2 = a2 * inv_a0;
}

static float biquad_process(wd_biquad_t *bq, float in) {
    float out = bq->b0 * in + bq->b1 * bq->x1 + bq->b2 * bq->x2
                             - bq->a1 * bq->y1 - bq->a2 * bq->y2;
    bq->x2 = bq->x1; bq->x1 = in;
    bq->y2 = bq->y1; bq->y1 = out;
    return out;
}

/* ── LPF/HPF biquad coefficient calculators (Octocosme Isolator3 pattern) ── */
static void biquad_set_lpf(wd_biquad_t *bq, float freq, float q) {
    float w0 = TWO_PI * clampf(freq, 20.0f, SAMPLE_RATE * 0.49f) / SAMPLE_RATE;
    float alpha = sinf(w0) / (2.0f * clampf(q, 0.1f, 20.0f));
    float cosw = cosf(w0);
    float a0_inv = 1.0f / (1.0f + alpha);
    bq->b0 = (1.0f - cosw) * 0.5f * a0_inv;
    bq->b1 = (1.0f - cosw) * a0_inv;
    bq->b2 = bq->b0;
    bq->a1 = -2.0f * cosw * a0_inv;
    bq->a2 = (1.0f - alpha) * a0_inv;
}

static void biquad_set_hpf(wd_biquad_t *bq, float freq, float q) {
    float w0 = TWO_PI * clampf(freq, 20.0f, SAMPLE_RATE * 0.49f) / SAMPLE_RATE;
    float alpha = sinf(w0) / (2.0f * clampf(q, 0.1f, 20.0f));
    float cosw = cosf(w0);
    float a0_inv = 1.0f / (1.0f + alpha);
    bq->b0 = (1.0f + cosw) * 0.5f * a0_inv;
    bq->b1 = -(1.0f + cosw) * a0_inv;
    bq->b2 = bq->b0;
    bq->a1 = -2.0f * cosw * a0_inv;
    bq->a2 = (1.0f - alpha) * a0_inv;
}

typedef struct {
    /* Dirty Compressor (0-50% = compress, 50-100% = compress+distort) */
    float comp_amount;      /* 0..1 */
    float comp_env;         /* envelope follower state */

    /* DJ Filter (Isolator3-style 3-stage cascade) */
    float dj_filter;        /* 0..1 (0=LP, 0.5=bypass, 1=HP) */
    float dj_freq_smooth;   /* smoothed cutoff */
    wd_biquad_t dj_lpf1, dj_lpf2, dj_lpf3;
    wd_biquad_t dj_hpf1, dj_hpf2, dj_hpf3;

    /* 3-band parametric EQ (Massenburg 8200 style) */
    float eq_low_gain;      /* -12..+12 dB */
    float eq_mid_gain;      /* -12..+12 dB */
    float eq_high_gain;     /* -12..+12 dB */
    float eq_low_freq;      /* 20..500 Hz */
    float eq_mid_freq;      /* 200..8000 Hz */
    float eq_high_freq;     /* 2000..18000 Hz */
    float eq_low_q;         /* 0.3..8.0 */
    float eq_mid_q;         /* 0.3..8.0 */
    float eq_high_q;        /* 0.3..8.0 */

    /* Biquad filter instances for EQ */
    wd_biquad_t eq_lo_bq;
    wd_biquad_t eq_mid_bq;
    wd_biquad_t eq_hi_bq;

    /* Coefficient update counter */
    int eq_update_counter;

    /* Master level */
    float master_level;     /* 0..1 */
} wd_master_t;

#define EQ_UPDATE_INTERVAL 64

static void master_init(wd_master_t *m) {
    memset(m, 0, sizeof(*m));
    m->comp_amount = 0.0f;
    m->dj_filter = 0.5f;  /* center = bypass */
    m->dj_freq_smooth = 0.5f;
    m->eq_low_gain = 0.0f;
    m->eq_mid_gain = 0.0f;
    m->eq_high_gain = 0.0f;
    m->eq_low_freq = 200.0f;
    m->eq_mid_freq = 1000.0f;
    m->eq_high_freq = 5000.0f;
    m->eq_low_q = 0.7f;
    m->eq_mid_q = 0.7f;
    m->eq_high_q = 0.7f;
    m->master_level = 0.8f;
    m->eq_update_counter = 0;

    biquad_set_peaking(&m->eq_lo_bq, 200.0f, 0.0f, 0.7f);
    biquad_set_peaking(&m->eq_mid_bq, 1000.0f, 0.0f, 0.7f);
    biquad_set_peaking(&m->eq_hi_bq, 5000.0f, 0.0f, 0.7f);
}

/* Block-rate update for DJ filter coefficients + EQ coefficients */
static void master_update_coeffs(wd_master_t *m) {
    /* DJ filter smoothing + coefficient update */
    m->dj_freq_smooth += 0.05f * (m->dj_filter - m->dj_freq_smooth);
    float fs = m->dj_freq_smooth;
    if (fs < 0.49f) {
        float t = (0.49f - fs) / 0.49f;
        float lp_f = 18000.0f * powf(200.0f / 18000.0f, t);
        biquad_set_lpf(&m->dj_lpf1, lp_f, 0.707f);
        biquad_set_lpf(&m->dj_lpf2, lp_f, 0.707f);
        biquad_set_lpf(&m->dj_lpf3, lp_f, 0.707f);
    } else if (fs > 0.51f) {
        float t = (fs - 0.51f) / 0.49f;
        float hp_f = 20.0f * powf(400.0f, t);
        biquad_set_hpf(&m->dj_hpf1, hp_f, 0.707f);
        biquad_set_hpf(&m->dj_hpf2, hp_f, 0.707f);
        biquad_set_hpf(&m->dj_hpf3, hp_f, 0.707f);
    }

    /* EQ coefficient update */
    biquad_set_peaking(&m->eq_lo_bq, m->eq_low_freq, m->eq_low_gain, m->eq_low_q);
    biquad_set_peaking(&m->eq_mid_bq, m->eq_mid_freq, m->eq_mid_gain, m->eq_mid_q);
    biquad_set_peaking(&m->eq_hi_bq, m->eq_high_freq, m->eq_high_gain, m->eq_high_q);
}

static float master_process(wd_master_t *m, float in) {
    float out = in;

    /* ── Dirty Compressor ──
     * 0-50%: clean compression (ratio 1:1 → 5:1)
     * 50-100%: compression + tanh saturation (increasingly destroyed) */
    if (m->comp_amount > 0.01f) {
        float abs_in = fabsf(out);
        /* Envelope follower: fast attack, slow release */
        if (abs_in > m->comp_env)
            m->comp_env += 0.01f * (abs_in - m->comp_env);
        else
            m->comp_env += 0.0001f * (abs_in - m->comp_env);

        /* Compression: lower threshold + higher ratio as amount increases */
        float threshold = 0.6f - m->comp_amount * 0.4f;  /* 0.6 → 0.2 */
        float ratio = 1.0f + m->comp_amount * 4.0f;      /* 1:1 → 5:1 */
        if (m->comp_env > threshold) {
            float over = m->comp_env - threshold;
            float target_over = over / ratio;
            float gain = (threshold + target_over) / (threshold + over);
            out *= gain;
        }

        /* Makeup gain: auto-compensate for gain reduction */
        float makeup = 1.0f + m->comp_amount * 0.4f;
        out *= makeup;

        /* Above 50%: add saturation that gets progressively filthier */
        if (m->comp_amount > 0.5f) {
            float dirt = (m->comp_amount - 0.5f) * 2.0f;  /* 0..1 over 50-100% */
            float drive = 1.0f + dirt * dirt * 8.0f;       /* 1x → 9x, quadratic */
            float wet = tanhf(out * drive);
            /* Crossfade: progressively more saturated signal */
            out = out * (1.0f - dirt) + wet * dirt;
        }
    }

    /* ── DJ Filter (Isolator3-style 3-stage cascade) ── */
    float filt = m->dj_freq_smooth;
    if (filt < 0.49f) {
        out = biquad_process(&m->dj_lpf1, out);
        out = biquad_process(&m->dj_lpf2, out);
        out = biquad_process(&m->dj_lpf3, out);
    } else if (filt > 0.51f) {
        out = biquad_process(&m->dj_hpf1, out);
        out = biquad_process(&m->dj_hpf2, out);
        out = biquad_process(&m->dj_hpf3, out);
    }

    /* ── 3-Band Parametric EQ (Massenburg 8200 style) ── */
    if (--m->eq_update_counter <= 0) {
        m->eq_update_counter = EQ_UPDATE_INTERVAL;
        master_update_coeffs(m);
    }
    out = biquad_process(&m->eq_lo_bq, out);
    out = biquad_process(&m->eq_mid_bq, out);
    out = biquad_process(&m->eq_hi_bq, out);

    /* ── Master Level ── */
    out *= m->master_level;

    return out;
}

/* ============================================================================
 * Instance
 * ============================================================================ */

/* ── 30 Kit presets: each defines 8 voice preset indices ── */
#define NUM_KITS 30
static const int KIT_PRESETS[NUM_KITS][8] = {
    /* 0  Default */     { 0, 5, 10, 25, 30, 15, 16, 20 },
    /* 1  808 */         { 0, 5, 10, 25, 31, 15, 16, 20 },
    /* 2  909 */         { 1, 6, 11, 26, 30, 15, 16, 20 },
    /* 3  Minimal */     { 1, 5, 10, 29, 32, 15, 16, 24 },
    /* 4  Industrial */  { 2, 8, 13, 28, 31, 18, 19, 21 },
    /* 5  Lo-Fi */       { 3, 7, 10, 27, 33, 19, 17, 23 },
    /* 6  Techno */      { 1, 6, 13, 25, 30, 15, 16, 20 },
    /* 7  House */       { 0, 5, 14, 25, 33, 15, 16, 20 },
    /* 8  Electro */     { 2, 9, 12, 26, 32, 18, 16, 21 },
    /* 9  Hip Hop */     { 0, 7, 14, 27, 30, 15, 16, 23 },
    /* 10 Trap */        { 0, 6, 11, 25, 30, 15, 16, 24 },
    /* 11 DnB */         { 1, 8, 12, 29, 34, 15, 16, 20 },
    /* 12 Dub */         { 4, 7, 10, 27, 31, 17, 16, 23 },
    /* 13 Ambient */     { 4, 9, 14, 27, 33, 17, 16, 24 },
    /* 14 Noise */       { 3, 9, 13, 28, 39, 19, 19, 36 },
    /* 15 Glitch */      { 2, 8, 13, 28, 35, 18, 19, 36 },
    /* 16 Perc Only */   { 30, 31, 32, 33, 34, 14, 11, 12 },
    /* 17 FX Only */     { 35, 36, 37, 38, 39, 35, 38, 37 },
    /* 18 All Kicks */   { 0, 1, 2, 3, 4, 0, 1, 2 },
    /* 19 All Snares */  { 5, 6, 7, 8, 9, 5, 6, 7 },
    /* 20 All Toms */    { 10, 11, 12, 13, 14, 10, 11, 12 },
    /* 21 All HH */      { 15, 16, 17, 18, 19, 15, 16, 17 },
    /* 22 All Cymbals */ { 20, 21, 22, 23, 24, 20, 21, 22 },
    /* 23 All Claps */   { 25, 26, 27, 28, 29, 25, 26, 27 },
    /* 24 Metal */       { 2, 8, 13, 28, 31, 18, 19, 22 },
    /* 25 Organic */     { 4, 7, 14, 27, 33, 17, 16, 23 },
    /* 26 Bright */      { 1, 6, 12, 26, 32, 15, 16, 21 },
    /* 27 Dark */        { 0, 7, 10, 27, 31, 19, 17, 23 },
    /* 28 Fast */        { 1, 6, 12, 26, 30, 15, 15, 38 },
    /* 29 Weird */       { 2, 8, 13, 28, 35, 18, 19, 36 },
};
static const char *KIT_NAMES[NUM_KITS] = {
    "Default", "808", "909", "Minimal", "Industrial",
    "Lo-Fi", "Techno", "House", "Electro", "Hip Hop",
    "Trap", "DnB", "Dub", "Ambient", "Noise",
    "Glitch", "Perc Only", "FX Only", "All Kicks", "All Snares",
    "All Toms", "All HH", "All Cymbal", "All Claps", "Metal",
    "Organic", "Bright", "Dark", "Fast", "Weird"
};

typedef struct {
    wd_voice_t  voice[NUM_VOICES];
    wd_master_t master;
    float       voice_vol[NUM_VOICES];   /* mixer page volumes */
    float       voice_vol_smooth[NUM_VOICES];
    float       voice_pan_smooth[NUM_VOICES]; /* smoothed pan */
    int         current_page;            /* 0=mixer, 1=general, 2=patch, 3=pan, 4..11=voice1..8 */
    int         midi_voice_cursor;       /* round-robin for MIDI trigger */
    int         current_kit;             /* 0..29 kit preset index */
    float       same_freq;               /* 0=off, >0 = master freq override (20..20000) */
    uint32_t    rng_state;               /* RNG for randomize */
} wd_instance_t;

/* Simple RNG for randomize */
static float inst_random(wd_instance_t *inst) {
    uint32_t x = inst->rng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    inst->rng_state = x;
    return (float)(x & 0xFFFF) / 65536.0f;
}

static void randomize_voice(wd_instance_t *inst, int vi) {
    wd_voice_t *v = &inst->voice[vi];
    v->freq = 20.0f * powf(1000.0f, inst_random(inst));
    v->attack = 0.0001f + inst_random(inst) * 0.05f;
    v->decay = 0.02f + inst_random(inst) * inst_random(inst) * 2.0f;
    v->wave = inst_random(inst);
    v->pitch_env_amt = inst_random(inst) * inst_random(inst);
    v->pitch_env_rate = 0.005f + inst_random(inst) * 0.2f;
    v->pitch_lfo_amt = inst_random(inst) < 0.3f ? inst_random(inst) * 0.2f : 0.0f;
    v->pitch_lfo_rate = 0.5f + inst_random(inst) * 60.0f;
    v->filter_type = (int)(inst_random(inst) * 3.0f) % 3;
    v->filter_cutoff = 200.0f * powf(90.0f, inst_random(inst));
    v->filter_res = 1.0f + inst_random(inst) * 3.0f;
    v->noise_attack = 0.0001f + inst_random(inst) * 0.01f;
    v->noise_decay = 0.01f + inst_random(inst) * inst_random(inst) * 0.5f;
    v->mix = inst_random(inst);
    v->distortion = inst_random(inst) * inst_random(inst) * 30.0f;
    v->level = 0.5f + inst_random(inst) * 0.4f;
    v->clap_count = inst_random(inst) < 0.2f ? (int)(inst_random(inst) * 4) : 0;
    v->clap_spacing = 300 + (int)(inst_random(inst) * 400);
    v->preset = NUM_PRESETS - 1; /* Custom */
}

static void randomize_patch(wd_instance_t *inst) {
    for (int i = 0; i < NUM_VOICES; i++) {
        randomize_voice(inst, i);
        inst->voice_vol[i] = 0.5f + inst_random(inst) * 0.4f;
    }
}

static void apply_kit(wd_instance_t *inst, int kit) {
    if (kit < 0 || kit >= NUM_KITS) return;
    inst->current_kit = kit;
    for (int i = 0; i < NUM_VOICES; i++) {
        voice_apply_preset(&inst->voice[i], KIT_PRESETS[kit][i]);
        inst->voice_vol[i] = inst->voice[i].level;
    }
}

/* ── MIDI note to voice mapping ──
 * Pads on Move send notes. We map pad positions to voices:
 *   Voice 0-7 triggered by notes: any note triggers round-robin,
 *   OR fixed mapping: notes 36-43 (C2-G#2) = voices 0-7
 */
#define MIDI_NOTE_BASE 36

/* ── Page/Knob mapping tables ── */

/* Mixer page: 8 volume knobs */
static const char *MIXER_KNOB_KEYS[8] = {
    "v1_vol", "v2_vol", "v3_vol", "v4_vol",
    "v5_vol", "v6_vol", "v7_vol", "v8_vol"
};
static const char *MIXER_KNOB_NAMES[8] = {
    "Kick", "Snare", "Tom", "Clap",
    "Rim", "HiHat", "Cymbal", "Tom2"
};

/* General page: crush, filter, 3-band EQ (gain+freq paired) */
static const char *GENERAL_KNOB_KEYS[8] = {
    "comp", "dj_filter", "eq_lo", "lo_freq",
    "eq_mid", "mid_freq", "eq_hi", "hi_freq"
};
static const char *GENERAL_KNOB_NAMES[8] = {
    "Crush", "Filter", "Lo Gain", "Lo Freq",
    "Mid Gain", "Mid Freq", "Hi Gain", "Hi Freq"
};

/* Per-voice page: freq, decay, wave, p.env amt, mix, cutoff, distort, preset */
static const char *VOICE_KNOB_SUFFIXES[8] = {
    "_freq", "_decay", "_wave", "_penv",
    "_mix", "_cutoff", "_dist", "_preset"
};
static const char *VOICE_KNOB_NAMES[8] = {
    "Freq", "Decay", "Wave", "P.Env",
    "Mix", "Cutoff", "Distort", "Preset"
};

static const char *PRESET_NAMES[NUM_PRESETS] = {
    /* Kicks 0-4 */     "Sub Kick", "Punch Kick", "FM Kick", "Dusty Kick", "Long Kick",
    /* Snares 5-9 */    "Snare", "Snap Snare", "Fat Snare", "Crack Snare", "Noise Snare",
    /* Toms 10-14 */    "Low Tom", "Mid Tom", "High Tom", "Acid Tom", "Conga",
    /* HH 15-19 */      "Closed HH", "Open HH", "Pedal HH", "Metal HH", "Trash HH",
    /* Cymbals 20-24 */ "Crash", "Ride", "Ride Bell", "Dark Crash", "Sizzle",
    /* Claps 25-29 */   "Clap 808", "Tight Clap", "Big Clap", "Dirty Clap", "Snap Clap",
    /* Perc 30-34 */    "Rimshot", "Cowbell", "Clave", "Shaker", "Tamb",
    /* FX 35-39 */      "Zap", "Glitch", "Drone Hit", "Blip", "Noise Burst",
    /* 40 */            "Custom"
};
static const char *FILTER_NAMES[3] = { "LP", "HP", "BP" };

/* ============================================================================
 * Parameter helpers
 * ============================================================================ */

/* Get voice knob key string: "v1_freq", "v3_decay", etc. */
static void voice_key(char *buf, int buflen, int voice_idx, const char *suffix) {
    snprintf(buf, buflen, "v%d%s", voice_idx + 1, suffix);
}

/* Scale a 0..1 knob value to a parameter range */
static float knob_to_freq(float k) {
    /* Exponential 20..20000 Hz */
    return 20.0f * powf(1000.0f, k);
}
static float freq_to_knob(float f) {
    return logf(f / 20.0f) / logf(1000.0f);
}
static float knob_to_cutoff(float k) {
    return 20.0f * powf(900.0f, k);
}
static float cutoff_to_knob(float f) {
    return logf(f / 20.0f) / logf(900.0f);
}
static float knob_to_decay(float k) {
    /* 0.0001 .. 4.0 exponential */
    return 0.0001f * powf(40000.0f, k);
}
static float decay_to_knob(float d) {
    return logf(d / 0.0001f) / logf(40000.0f);
}
static float knob_to_eq_db(float k) {
    /* 0..1 -> -12..+12 dB */
    return (k - 0.5f) * 24.0f;
}
static float eq_db_to_knob(float db) {
    return db / 24.0f + 0.5f;
}
static float knob_to_lo_freq(float k) {
    return 20.0f + k * 480.0f; /* 20..500 */
}
static float lo_freq_to_knob(float f) {
    return (f - 20.0f) / 480.0f;
}
static float knob_to_mid_freq(float k) {
    return 200.0f * powf(40.0f, k); /* 200..8000 */
}
static float mid_freq_to_knob(float f) {
    return logf(f / 200.0f) / logf(40.0f);
}
static float knob_to_dist(float k) {
    return k * 50.0f; /* 0..50 dB */
}
static float dist_to_knob(float d) {
    return d / 50.0f;
}

/* ============================================================================
 * Plugin API
 * ============================================================================ */

static void *create_instance(const char *module_dir, const char *json_defaults) {
    (void)module_dir; (void)json_defaults;

    wd_instance_t *inst = calloc(1, sizeof(wd_instance_t));
    if (!inst) return NULL;

    for (int i = 0; i < NUM_VOICES; i++) {
        voice_init(&inst->voice[i], i);
        inst->voice_vol[i] = inst->voice[i].level;
        inst->voice_vol_smooth[i] = inst->voice_vol[i];
    }

    master_init(&inst->master);
    inst->current_page = 0;
    inst->midi_voice_cursor = 0;
    inst->current_kit = 0;
    inst->same_freq = 0.0f;
    inst->rng_state = 987654321u;

    return inst;
}

static void destroy_instance(void *instance) {
    free(instance);
}

static void on_midi(void *instance, const uint8_t *msg, int len, int source) {
    wd_instance_t *inst = (wd_instance_t *)instance;
    (void)source;
    if (!inst || len < 3) return;

    uint8_t status = msg[0] & 0xF0;
    uint8_t note   = msg[1];
    uint8_t vel    = msg[2];

    if (status == 0x90 && vel > 0) {
        /* Note On — map to voice */
        int voice_idx;
        if (note >= MIDI_NOTE_BASE && note < MIDI_NOTE_BASE + NUM_VOICES) {
            /* Fixed mapping: C2=voice0, C#2=voice1, ... G#2=voice7 */
            voice_idx = note - MIDI_NOTE_BASE;
        } else {
            /* Round-robin for other notes */
            voice_idx = inst->midi_voice_cursor;
            inst->midi_voice_cursor = (inst->midi_voice_cursor + 1) % NUM_VOICES;
        }
        voice_trigger(&inst->voice[voice_idx], (float)vel / 127.0f);

        /* Pad selects voice page for knob editing */
        inst->current_page = 4 + voice_idx;
    }
}

/* ── set_param ── */
static void set_param(void *instance, const char *key, const char *val) {
    wd_instance_t *inst = (wd_instance_t *)instance;
    if (!inst || !key || !val) return;

    /* Page switching: 0=mixer, 1=general, 2=patch, 3=pan, 4..11=voice1..8 */
    if (strcmp(key, "_level") == 0) {
        if (strcmp(val, "Mixer") == 0) inst->current_page = 0;
        else if (strcmp(val, "General") == 0) inst->current_page = 1;
        else if (strcmp(val, "Patch") == 0) inst->current_page = 2;
        else if (strcmp(val, "Pan") == 0) inst->current_page = 3;
        else {
            for (int i = 0; i < NUM_VOICES; i++) {
                char pg[16];
                snprintf(pg, sizeof(pg), "Voice %d", i + 1);
                if (strcmp(val, pg) == 0) { inst->current_page = 4 + i; break; }
            }
        }
        return;
    }

    /* Knob adjust */
    if (strncmp(key, "knob_", 5) == 0 && strstr(key, "_adjust")) {
        int knob = atoi(key + 5) - 1;
        if (knob < 0 || knob > 7) return;
        int delta = atoi(val);
        int page = inst->current_page;

        if (page == 0) {
            /* Mixer page: adjust voice volumes */
            inst->voice_vol[knob] = clampf(inst->voice_vol[knob] + delta * 0.01f, 0.0f, 1.0f);
        } else if (page == 1) {
            /* General page: comp, dist, lo gain+freq, mid gain+freq, hi gain+freq */
            float step = 0.01f;
            switch (knob) {
                case 0: inst->master.comp_amount = clampf(inst->master.comp_amount + delta * step, 0.0f, 1.0f); break;
                case 1: inst->master.dj_filter = clampf(inst->master.dj_filter + delta * 0.005f, 0.0f, 1.0f); break;
                case 2: inst->master.eq_low_gain = clampf(inst->master.eq_low_gain + delta * 0.24f, -12.0f, 12.0f); break;
                case 3: inst->master.eq_low_freq = clampf(inst->master.eq_low_freq + delta * 4.8f, 20.0f, 500.0f); break;
                case 4: inst->master.eq_mid_gain = clampf(inst->master.eq_mid_gain + delta * 0.24f, -12.0f, 12.0f); break;
                case 5: inst->master.eq_mid_freq = clampf(inst->master.eq_mid_freq + delta * 80.0f, 200.0f, 8000.0f); break;
                case 6: inst->master.eq_high_gain = clampf(inst->master.eq_high_gain + delta * 0.24f, -12.0f, 12.0f); break;
                case 7: inst->master.eq_high_freq = clampf(inst->master.eq_high_freq + delta * 160.0f, 2000.0f, 18000.0f); break;
            }
        } else if (page == 2) {
            /* Patch page: Kit, Rnd Voice, Rnd Patch, SameFreq, RndPan + 3 unused */
            switch (knob) {
                case 0: { /* Kit (jog) */
                    inst->current_kit = (inst->current_kit + (delta > 0 ? 1 : -1) + NUM_KITS) % NUM_KITS;
                    apply_kit(inst, inst->current_kit);
                } break;
                case 1: /* Rnd Voice — randomize the last-played voice */
                    if (delta != 0) {
                        int vi = inst->current_page >= 4 ? inst->current_page - 4 : 0;
                        randomize_voice(inst, vi);
                    } break;
                case 2: /* Rnd Patch — randomize entire kit */
                    if (delta != 0) randomize_patch(inst);
                    break;
                case 3: { /* SameFreq — master frequency + filter cutoff for all voices */
                    float k = inst->same_freq > 0.0f ? freq_to_knob(inst->same_freq) : 0.5f;
                    k = clampf(k + delta * 0.005f, 0.0f, 1.0f);
                    inst->same_freq = knob_to_freq(k);
                    for (int i = 0; i < NUM_VOICES; i++) {
                        inst->voice[i].freq = inst->same_freq;
                        inst->voice[i].filter_cutoff = clampf(inst->same_freq, 20.0f, 18000.0f);
                    }
                } break;
                case 4: /* Rnd Pan — randomize panning (kicks stay center) */
                    if (delta != 0) {
                        for (int i = 0; i < NUM_VOICES; i++) {
                            if (inst->voice[i].preset <= 4) /* Kicks (0-4) stay center */
                                inst->voice[i].pan = 0.0f;
                            else
                                inst->voice[i].pan = (inst_random(inst) * 2.0f - 1.0f) * 0.8f;
                        }
                    } break;
            }
        } else if (page == 3) {
            /* Pan page: 8 pan knobs */
            if (knob >= 0 && knob < NUM_VOICES)
                inst->voice[knob].pan = clampf(inst->voice[knob].pan + delta * 0.02f, -1.0f, 1.0f);
        } else {
            /* Voice pages (4..11) */
            int vi = page - 4;
            if (vi < 0 || vi >= NUM_VOICES) return;
            wd_voice_t *v = &inst->voice[vi];
            switch (knob) {
                case 0: { /* Freq (exponential) */
                    float k = freq_to_knob(v->freq);
                    k = clampf(k + delta * 0.005f, 0.0f, 1.0f);
                    v->freq = knob_to_freq(k);
                } break;
                case 1: { /* Decay (exponential) */
                    float k = decay_to_knob(v->decay);
                    k = clampf(k + delta * 0.01f, 0.0f, 1.0f);
                    v->decay = knob_to_decay(k);
                } break;
                case 2: /* Wave (continuous morph) */
                    v->wave = clampf(v->wave + delta * 0.01f, 0.0f, 1.0f);
                    break;
                case 3: /* P.Env Amount */
                    v->pitch_env_amt = clampf(v->pitch_env_amt + delta * 0.01f, 0.0f, 1.0f);
                    break;
                case 4: /* Mix */
                    v->mix = clampf(v->mix + delta * 0.01f, 0.0f, 1.0f);
                    break;
                case 5: { /* Cutoff (exponential) */
                    float k = cutoff_to_knob(v->filter_cutoff);
                    k = clampf(k + delta * 0.005f, 0.0f, 1.0f);
                    v->filter_cutoff = knob_to_cutoff(k);
                } break;
                case 6: /* Distortion */
                    v->distortion = clampf(v->distortion + delta * 0.5f, 0.0f, 50.0f);
                    break;
                case 7: { /* Preset (jog) */
                    v->preset = (v->preset + (delta > 0 ? 1 : -1) + NUM_PRESETS) % NUM_PRESETS;
                    if (v->preset < NUM_PRESETS - 1) voice_apply_preset(v, v->preset);
                } break;
            }
        }
        return;
    }

    /* Direct parameter set (state restore, presets) */
    float f = atof(val);

    /* Mixer volumes */
    for (int i = 0; i < NUM_VOICES; i++) {
        char k[16];
        snprintf(k, sizeof(k), "v%d_vol", i + 1);
        if (strcmp(key, k) == 0) { inst->voice_vol[i] = clampf(f, 0.0f, 1.0f); return; }
    }

    /* Master params */
    if (strcmp(key, "comp") == 0) { inst->master.comp_amount = clampf(f, 0.0f, 1.0f); return; }
    if (strcmp(key, "dj_filter") == 0) { inst->master.dj_filter = clampf(f, 0.0f, 1.0f); return; }
    if (strcmp(key, "eq_lo") == 0) { inst->master.eq_low_gain = clampf(f, -12.0f, 12.0f); return; }
    if (strcmp(key, "eq_mid") == 0) { inst->master.eq_mid_gain = clampf(f, -12.0f, 12.0f); return; }
    if (strcmp(key, "eq_hi") == 0) { inst->master.eq_high_gain = clampf(f, -12.0f, 12.0f); return; }
    if (strcmp(key, "lo_freq") == 0) { inst->master.eq_low_freq = clampf(f, 20.0f, 500.0f); return; }
    if (strcmp(key, "mid_freq") == 0) { inst->master.eq_mid_freq = clampf(f, 200.0f, 8000.0f); return; }
    if (strcmp(key, "hi_freq") == 0) { inst->master.eq_high_freq = clampf(f, 2000.0f, 18000.0f); return; }
    if (strcmp(key, "q_lo") == 0) { inst->master.eq_low_q = clampf(f, 0.3f, 8.0f); return; }
    if (strcmp(key, "q_mid") == 0) { inst->master.eq_mid_q = clampf(f, 0.3f, 8.0f); return; }
    if (strcmp(key, "q_hi") == 0) { inst->master.eq_high_q = clampf(f, 0.3f, 8.0f); return; }
    if (strcmp(key, "kit") == 0) { apply_kit(inst, (int)clampf(f, 0, NUM_KITS-1)); return; }
    if (strcmp(key, "rnd_voice") == 0) { if (f != 0) { int vi = inst->current_page >= 3 ? inst->current_page - 3 : 0; randomize_voice(inst, vi); } return; }
    if (strcmp(key, "rnd_patch") == 0) { if (f != 0) randomize_patch(inst); return; }
    if (strcmp(key, "rnd_pan") == 0) {
        if (f != 0) { for (int i=0;i<NUM_VOICES;i++) { if (inst->voice[i].preset<=4) inst->voice[i].pan=0; else inst->voice[i].pan=(inst_random(inst)*2.0f-1.0f)*0.8f; } }
        return;
    }
    if (strcmp(key, "same_freq") == 0) { inst->same_freq = clampf(f, 20.0f, 20000.0f); for (int i=0;i<NUM_VOICES;i++) { inst->voice[i].freq = inst->same_freq; inst->voice[i].filter_cutoff = clampf(inst->same_freq, 20.0f, 18000.0f); } return; }
    if (strcmp(key, "master") == 0) { inst->master.master_level = clampf(f, 0.0f, 1.0f); return; }

    /* Per-voice params: v1_freq, v2_decay, etc. */
    for (int i = 0; i < NUM_VOICES; i++) {
        char k[24];
        wd_voice_t *v = &inst->voice[i];

        snprintf(k, sizeof(k), "v%d_freq", i+1);
        if (strcmp(key, k) == 0) { v->freq = clampf(f, 20.0f, 20000.0f); return; }
        snprintf(k, sizeof(k), "v%d_decay", i+1);
        if (strcmp(key, k) == 0) { v->decay = clampf(f, 0.0001f, 4.0f); return; }
        snprintf(k, sizeof(k), "v%d_wave", i+1);
        if (strcmp(key, k) == 0) {
            /* Accept string names or float 0..1 */
            if (strcmp(val, "Sine") == 0) v->wave = 0.0f;
            else if (strcmp(val, "Triangle") == 0) v->wave = 0.33f;
            else if (strcmp(val, "Saw") == 0) v->wave = 0.66f;
            else if (strcmp(val, "Square") == 0) v->wave = 1.0f;
            else v->wave = clampf(f, 0.0f, 1.0f);
            return;
        }
        snprintf(k, sizeof(k), "v%d_penv", i+1);
        if (strcmp(key, k) == 0) { v->pitch_env_amt = clampf(f, 0.0f, 1.0f); return; }
        snprintf(k, sizeof(k), "v%d_mix", i+1);
        if (strcmp(key, k) == 0) { v->mix = clampf(f, 0.0f, 1.0f); return; }
        snprintf(k, sizeof(k), "v%d_cutoff", i+1);
        if (strcmp(key, k) == 0) { v->filter_cutoff = clampf(f, 20.0f, 18000.0f); return; }
        snprintf(k, sizeof(k), "v%d_dist", i+1);
        if (strcmp(key, k) == 0) { v->distortion = clampf(f, 0.0f, 50.0f); return; }
        snprintf(k, sizeof(k), "v%d_preset", i+1);
        if (strcmp(key, k) == 0) {
            /* Accept string names or numeric */
            int p = -1;
            for (int j = 0; j < 8; j++) {
                if (strcmp(val, PRESET_NAMES[j]) == 0) { p = j; break; }
            }
            if (p < 0) p = (int)f;
            if (p >= 0 && p < NUM_PRESETS - 1) voice_apply_preset(v, p);
            else v->preset = NUM_PRESETS - 1;
            return;
        }
        snprintf(k, sizeof(k), "v%d_attack", i+1);
        if (strcmp(key, k) == 0) { v->attack = clampf(f, 0.0001f, 1.0f); return; }
        snprintf(k, sizeof(k), "v%d_ndecay", i+1);
        if (strcmp(key, k) == 0) { v->noise_decay = clampf(f, 0.0001f, 1.0f); return; }
        snprintf(k, sizeof(k), "v%d_ftype", i+1);
        if (strcmp(key, k) == 0) {
            if (strcmp(val, "LP") == 0) v->filter_type = 0;
            else if (strcmp(val, "HP") == 0) v->filter_type = 1;
            else if (strcmp(val, "BP") == 0) v->filter_type = 2;
            else v->filter_type = (int)clampf(f, 0, 2);
            return;
        }
        snprintf(k, sizeof(k), "v%d_fres", i+1);
        if (strcmp(key, k) == 0) { v->filter_res = clampf(f, 1.0f, 5.0f); return; }
        snprintf(k, sizeof(k), "v%d_prate", i+1);
        if (strcmp(key, k) == 0) { v->pitch_env_rate = clampf(f, 0.001f, 2.0f); return; }
        snprintf(k, sizeof(k), "v%d_lamt", i+1);
        if (strcmp(key, k) == 0) { v->pitch_lfo_amt = clampf(f, 0.0f, 1.0f); return; }
        snprintf(k, sizeof(k), "v%d_lrate", i+1);
        if (strcmp(key, k) == 0) { v->pitch_lfo_rate = clampf(f, 0.1f, 80.0f); return; }
        snprintf(k, sizeof(k), "v%d_nattack", i+1);
        if (strcmp(key, k) == 0) { v->noise_attack = clampf(f, 0.0001f, 1.0f); return; }
        snprintf(k, sizeof(k), "v%d_level", i+1);
        if (strcmp(key, k) == 0) { v->level = clampf(f, 0.0f, 1.0f); return; }
        snprintf(k, sizeof(k), "v%d_pan", i+1);
        if (strcmp(key, k) == 0) { v->pan = clampf(f, -1.0f, 1.0f); return; }
    }

    /* State restore (all params in one string) */
    if (strcmp(key, "state") == 0) {
        /* Parse: 8 voice volumes, master params, then per-voice params */
        const char *p = val;
        char token[64];
        int ti = 0;

        /* Helper: read next space-delimited token */
        #define NEXT_TOKEN() do { \
            while (*p == ' ') p++; \
            ti = 0; \
            while (*p && *p != ' ' && ti < 63) token[ti++] = *p++; \
            token[ti] = '\0'; \
        } while(0)

        /* 8 mixer volumes */
        for (int i = 0; i < NUM_VOICES; i++) {
            NEXT_TOKEN();
            inst->voice_vol[i] = clampf(atof(token), 0.0f, 1.0f);
        }
        /* Master: comp, dist, eq_lo, eq_mid, eq_hi, lo_freq, mid_freq, hi_freq, master */
        NEXT_TOKEN(); inst->master.comp_amount = atof(token);
        NEXT_TOKEN(); inst->master.dj_filter = atof(token);
        NEXT_TOKEN(); inst->master.eq_low_gain = atof(token);
        NEXT_TOKEN(); inst->master.eq_mid_gain = atof(token);
        NEXT_TOKEN(); inst->master.eq_high_gain = atof(token);
        NEXT_TOKEN(); inst->master.eq_low_freq = atof(token);
        NEXT_TOKEN(); inst->master.eq_mid_freq = atof(token);
        NEXT_TOKEN(); inst->master.eq_high_freq = atof(token);
        NEXT_TOKEN(); inst->master.eq_low_q = atof(token);
        NEXT_TOKEN(); inst->master.eq_mid_q = atof(token);
        NEXT_TOKEN(); inst->master.eq_high_q = atof(token);
        NEXT_TOKEN(); inst->master.dj_filter = atof(token);
        NEXT_TOKEN(); inst->master.master_level = atof(token);

        /* Per-voice: preset freq attack decay wave penv prate lamt lrate ftype cutoff fres nattack ndecay mix dist level */
        for (int i = 0; i < NUM_VOICES; i++) {
            wd_voice_t *v = &inst->voice[i];
            NEXT_TOKEN(); v->preset = atoi(token);
            NEXT_TOKEN(); v->freq = atof(token);
            NEXT_TOKEN(); v->attack = atof(token);
            NEXT_TOKEN(); v->decay = atof(token);
            NEXT_TOKEN(); v->wave = atof(token);
            NEXT_TOKEN(); v->pitch_env_amt = atof(token);
            NEXT_TOKEN(); v->pitch_env_rate = atof(token);
            NEXT_TOKEN(); v->pitch_lfo_amt = atof(token);
            NEXT_TOKEN(); v->pitch_lfo_rate = atof(token);
            NEXT_TOKEN(); v->filter_type = atoi(token);
            NEXT_TOKEN(); v->filter_cutoff = atof(token);
            NEXT_TOKEN(); v->filter_res = atof(token);
            NEXT_TOKEN(); v->noise_attack = atof(token);
            NEXT_TOKEN(); v->noise_decay = atof(token);
            NEXT_TOKEN(); v->mix = atof(token);
            NEXT_TOKEN(); v->distortion = atof(token);
            NEXT_TOKEN(); v->level = atof(token);
            NEXT_TOKEN(); v->pan = atof(token);
        }
        #undef NEXT_TOKEN
        return;
    }
}

/* ── get_param ── */
static int get_param(void *instance, const char *key, char *buf, int buf_len) {
    wd_instance_t *inst = (wd_instance_t *)instance;
    if (!inst || !key || !buf || buf_len < 1) return -1;

    /* Module name */
    if (strcmp(key, "name") == 0)
        return snprintf(buf, buf_len, "WDrm");

    /* Knob names */
    if (strncmp(key, "knob_", 5) == 0 && strstr(key, "_name")) {
        int knob = atoi(key + 5) - 1;
        if (knob < 0 || knob > 7) return -1;
        int page = inst->current_page;

        if (page == 0) return snprintf(buf, buf_len, "%s", MIXER_KNOB_NAMES[knob]);
        if (page == 1) return snprintf(buf, buf_len, "%s", GENERAL_KNOB_NAMES[knob]);
        if (page == 2) {
            static const char *PATCH_N[8] = {"Kit","Rnd Voice","Rnd Patch","SameFreq","Rnd Pan","","",""};
            return snprintf(buf, buf_len, "%s", PATCH_N[knob]);
        }
        if (page == 3) return snprintf(buf, buf_len, "%s", MIXER_KNOB_NAMES[knob]);
        /* Voice pages */
        return snprintf(buf, buf_len, "%s", VOICE_KNOB_NAMES[knob]);
    }

    /* Knob values */
    if (strncmp(key, "knob_", 5) == 0 && strstr(key, "_value")) {
        int knob = atoi(key + 5) - 1;
        if (knob < 0 || knob > 7) return -1;
        int page = inst->current_page;

        if (page == 0) {
            return snprintf(buf, buf_len, "%d%%", (int)(inst->voice_vol[knob] * 100.0f));
        }
        if (page == 1) {
            switch (knob) {
                case 0: return snprintf(buf, buf_len, "%d%%", (int)(inst->master.comp_amount * 100.0f));
                case 1: {
                    float f = inst->master.dj_filter;
                    if (f < 0.49f) return snprintf(buf, buf_len, "LP %d%%", (int)((0.49f - f) / 0.49f * 100.0f));
                    if (f > 0.51f) return snprintf(buf, buf_len, "HP %d%%", (int)((f - 0.51f) / 0.49f * 100.0f));
                    return snprintf(buf, buf_len, "Off");
                }
                case 2: return snprintf(buf, buf_len, "%+.0fdB", inst->master.eq_low_gain);
                case 3: return snprintf(buf, buf_len, "%dHz", (int)inst->master.eq_low_freq);
                case 4: return snprintf(buf, buf_len, "%+.0fdB", inst->master.eq_mid_gain);
                case 5: return snprintf(buf, buf_len, "%dHz", (int)inst->master.eq_mid_freq);
                case 6: return snprintf(buf, buf_len, "%+.0fdB", inst->master.eq_high_gain);
                case 7: return snprintf(buf, buf_len, "%dHz", (int)inst->master.eq_high_freq);
            }
        }
        if (page == 2) {
            switch (knob) {
                case 0: return snprintf(buf, buf_len, "%s", KIT_NAMES[inst->current_kit]);
                case 1: return snprintf(buf, buf_len, "Turn");
                case 2: return snprintf(buf, buf_len, "Turn");
                case 3: return snprintf(buf, buf_len, "%dHz", inst->same_freq > 0 ? (int)inst->same_freq : 0);
                case 4: return snprintf(buf, buf_len, "Turn");
                default: return snprintf(buf, buf_len, "-");
            }
        }
        if (page == 3) {
            /* Pan page */
            if (knob < NUM_VOICES) {
                float p = inst->voice[knob].pan;
                if (p < -0.01f) return snprintf(buf, buf_len, "L%d", (int)(-p * 100));
                if (p > 0.01f) return snprintf(buf, buf_len, "R%d", (int)(p * 100));
                return snprintf(buf, buf_len, "C");
            }
        }
        /* Voice page */
        int vi = page - 4;
        if (vi < 0 || vi >= NUM_VOICES) return -1;
        wd_voice_t *v = &inst->voice[vi];
        switch (knob) {
            case 0: return snprintf(buf, buf_len, "%dHz", (int)v->freq);
            case 1: return snprintf(buf, buf_len, "%dms", (int)(v->decay * 1000.0f));
            case 2: {
                    float w = v->wave * 3.0f;
                    if (w < 0.1f) return snprintf(buf, buf_len, "Sine");
                    if (w < 1.0f) return snprintf(buf, buf_len, "Si>Tri %d%%", (int)(w * 100));
                    if (w < 1.1f) return snprintf(buf, buf_len, "Tri");
                    if (w < 2.0f) return snprintf(buf, buf_len, "Tr>Saw %d%%", (int)((w-1)*100));
                    if (w < 2.1f) return snprintf(buf, buf_len, "Saw");
                    if (w < 3.0f) return snprintf(buf, buf_len, "Sw>Sqr %d%%", (int)((w-2)*100));
                    return snprintf(buf, buf_len, "Square");
                }
            case 3: return snprintf(buf, buf_len, "%d%%", (int)(v->pitch_env_amt * 100.0f));
            case 4: return snprintf(buf, buf_len, "%d%%", (int)(v->mix * 100.0f));
            case 5: return snprintf(buf, buf_len, "%dHz", (int)v->filter_cutoff);
            case 6: return snprintf(buf, buf_len, "%.0fdB", v->distortion);
            case 7: return snprintf(buf, buf_len, "%s", PRESET_NAMES[v->preset]);
        }
    }

    /* chain_params — memcpy pattern to avoid truncation */
    if (strcmp(key, "chain_params") == 0) {
        static const char *cp =
            "["
            "{\"key\":\"v1_vol\",\"name\":\"V1 Vol\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v2_vol\",\"name\":\"V2 Vol\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v3_vol\",\"name\":\"V3 Vol\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v4_vol\",\"name\":\"V4 Vol\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v5_vol\",\"name\":\"V5 Vol\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v6_vol\",\"name\":\"V6 Vol\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v7_vol\",\"name\":\"V7 Vol\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v8_vol\",\"name\":\"V8 Vol\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"comp\",\"name\":\"Compress\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"dj_filter\",\"name\":\"Filter\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.005},"
            "{\"key\":\"eq_lo\",\"name\":\"EQ Low\",\"type\":\"float\",\"min\":-12,\"max\":12,\"step\":0.24},"
            "{\"key\":\"eq_mid\",\"name\":\"EQ Mid\",\"type\":\"float\",\"min\":-12,\"max\":12,\"step\":0.24},"
            "{\"key\":\"eq_hi\",\"name\":\"EQ High\",\"type\":\"float\",\"min\":-12,\"max\":12,\"step\":0.24},"
            "{\"key\":\"lo_freq\",\"name\":\"Lo Freq\",\"type\":\"int\",\"min\":20,\"max\":500,\"step\":5},"
            "{\"key\":\"mid_freq\",\"name\":\"Mid Freq\",\"type\":\"int\",\"min\":200,\"max\":8000,\"step\":80},"
            "{\"key\":\"hi_freq\",\"name\":\"Hi Freq\",\"type\":\"int\",\"min\":2000,\"max\":18000,\"step\":160},"
            "{\"key\":\"q_lo\",\"name\":\"Lo Q\",\"type\":\"float\",\"min\":0.3,\"max\":8,\"step\":0.1},"
            "{\"key\":\"q_mid\",\"name\":\"Mid Q\",\"type\":\"float\",\"min\":0.3,\"max\":8,\"step\":0.1},"
            "{\"key\":\"q_hi\",\"name\":\"Hi Q\",\"type\":\"float\",\"min\":0.3,\"max\":8,\"step\":0.1},"
            "{\"key\":\"kit\",\"name\":\"Kit\",\"type\":\"int\",\"min\":0,\"max\":29,\"step\":1},"
            "{\"key\":\"same_freq\",\"name\":\"SameFreq\",\"type\":\"int\",\"min\":20,\"max\":20000,\"step\":1},"
            "{\"key\":\"master\",\"name\":\"Master\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v1_freq\",\"name\":\"V1 Freq\",\"type\":\"int\",\"min\":20,\"max\":20000,\"step\":1},"
            "{\"key\":\"v1_decay\",\"name\":\"V1 Decay\",\"type\":\"float\",\"min\":0,\"max\":4,\"step\":0.01},"
            "{\"key\":\"v1_wave\",\"name\":\"V1 Wave\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v1_penv\",\"name\":\"V1 P.Env\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v1_mix\",\"name\":\"V1 Mix\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v1_cutoff\",\"name\":\"V1 Cutoff\",\"type\":\"int\",\"min\":20,\"max\":18000,\"step\":1},"
            "{\"key\":\"v1_dist\",\"name\":\"V1 Distort\",\"type\":\"float\",\"min\":0,\"max\":50,\"step\":0.5},"
            "{\"key\":\"v1_preset\",\"name\":\"V1 Preset\",\"type\":\"int\",\"min\":0,\"max\":40,\"step\":1},"
            "{\"key\":\"v2_freq\",\"name\":\"V2 Freq\",\"type\":\"int\",\"min\":20,\"max\":20000,\"step\":1},"
            "{\"key\":\"v2_decay\",\"name\":\"V2 Decay\",\"type\":\"float\",\"min\":0,\"max\":4,\"step\":0.01},"
            "{\"key\":\"v2_wave\",\"name\":\"V2 Wave\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v2_penv\",\"name\":\"V2 P.Env\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v2_mix\",\"name\":\"V2 Mix\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v2_cutoff\",\"name\":\"V2 Cutoff\",\"type\":\"int\",\"min\":20,\"max\":18000,\"step\":1},"
            "{\"key\":\"v2_dist\",\"name\":\"V2 Distort\",\"type\":\"float\",\"min\":0,\"max\":50,\"step\":0.5},"
            "{\"key\":\"v2_preset\",\"name\":\"V2 Preset\",\"type\":\"int\",\"min\":0,\"max\":40,\"step\":1},"
            "{\"key\":\"v3_freq\",\"name\":\"V3 Freq\",\"type\":\"int\",\"min\":20,\"max\":20000,\"step\":1},"
            "{\"key\":\"v3_decay\",\"name\":\"V3 Decay\",\"type\":\"float\",\"min\":0,\"max\":4,\"step\":0.01},"
            "{\"key\":\"v3_wave\",\"name\":\"V3 Wave\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v3_penv\",\"name\":\"V3 P.Env\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v3_mix\",\"name\":\"V3 Mix\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v3_cutoff\",\"name\":\"V3 Cutoff\",\"type\":\"int\",\"min\":20,\"max\":18000,\"step\":1},"
            "{\"key\":\"v3_dist\",\"name\":\"V3 Distort\",\"type\":\"float\",\"min\":0,\"max\":50,\"step\":0.5},"
            "{\"key\":\"v3_preset\",\"name\":\"V3 Preset\",\"type\":\"int\",\"min\":0,\"max\":40,\"step\":1},"
            "{\"key\":\"v4_freq\",\"name\":\"V4 Freq\",\"type\":\"int\",\"min\":20,\"max\":20000,\"step\":1},"
            "{\"key\":\"v4_decay\",\"name\":\"V4 Decay\",\"type\":\"float\",\"min\":0,\"max\":4,\"step\":0.01},"
            "{\"key\":\"v4_wave\",\"name\":\"V4 Wave\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v4_penv\",\"name\":\"V4 P.Env\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v4_mix\",\"name\":\"V4 Mix\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v4_cutoff\",\"name\":\"V4 Cutoff\",\"type\":\"int\",\"min\":20,\"max\":18000,\"step\":1},"
            "{\"key\":\"v4_dist\",\"name\":\"V4 Distort\",\"type\":\"float\",\"min\":0,\"max\":50,\"step\":0.5},"
            "{\"key\":\"v4_preset\",\"name\":\"V4 Preset\",\"type\":\"int\",\"min\":0,\"max\":40,\"step\":1},"
            "{\"key\":\"v5_freq\",\"name\":\"V5 Freq\",\"type\":\"int\",\"min\":20,\"max\":20000,\"step\":1},"
            "{\"key\":\"v5_decay\",\"name\":\"V5 Decay\",\"type\":\"float\",\"min\":0,\"max\":4,\"step\":0.01},"
            "{\"key\":\"v5_wave\",\"name\":\"V5 Wave\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v5_penv\",\"name\":\"V5 P.Env\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v5_mix\",\"name\":\"V5 Mix\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v5_cutoff\",\"name\":\"V5 Cutoff\",\"type\":\"int\",\"min\":20,\"max\":18000,\"step\":1},"
            "{\"key\":\"v5_dist\",\"name\":\"V5 Distort\",\"type\":\"float\",\"min\":0,\"max\":50,\"step\":0.5},"
            "{\"key\":\"v5_preset\",\"name\":\"V5 Preset\",\"type\":\"int\",\"min\":0,\"max\":40,\"step\":1},"
            "{\"key\":\"v6_freq\",\"name\":\"V6 Freq\",\"type\":\"int\",\"min\":20,\"max\":20000,\"step\":1},"
            "{\"key\":\"v6_decay\",\"name\":\"V6 Decay\",\"type\":\"float\",\"min\":0,\"max\":4,\"step\":0.01},"
            "{\"key\":\"v6_wave\",\"name\":\"V6 Wave\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v6_penv\",\"name\":\"V6 P.Env\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v6_mix\",\"name\":\"V6 Mix\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v6_cutoff\",\"name\":\"V6 Cutoff\",\"type\":\"int\",\"min\":20,\"max\":18000,\"step\":1},"
            "{\"key\":\"v6_dist\",\"name\":\"V6 Distort\",\"type\":\"float\",\"min\":0,\"max\":50,\"step\":0.5},"
            "{\"key\":\"v6_preset\",\"name\":\"V6 Preset\",\"type\":\"int\",\"min\":0,\"max\":40,\"step\":1},"
            "{\"key\":\"v7_freq\",\"name\":\"V7 Freq\",\"type\":\"int\",\"min\":20,\"max\":20000,\"step\":1},"
            "{\"key\":\"v7_decay\",\"name\":\"V7 Decay\",\"type\":\"float\",\"min\":0,\"max\":4,\"step\":0.01},"
            "{\"key\":\"v7_wave\",\"name\":\"V7 Wave\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v7_penv\",\"name\":\"V7 P.Env\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v7_mix\",\"name\":\"V7 Mix\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v7_cutoff\",\"name\":\"V7 Cutoff\",\"type\":\"int\",\"min\":20,\"max\":18000,\"step\":1},"
            "{\"key\":\"v7_dist\",\"name\":\"V7 Distort\",\"type\":\"float\",\"min\":0,\"max\":50,\"step\":0.5},"
            "{\"key\":\"v7_preset\",\"name\":\"V7 Preset\",\"type\":\"int\",\"min\":0,\"max\":40,\"step\":1},"
            "{\"key\":\"v8_freq\",\"name\":\"V8 Freq\",\"type\":\"int\",\"min\":20,\"max\":20000,\"step\":1},"
            "{\"key\":\"v8_decay\",\"name\":\"V8 Decay\",\"type\":\"float\",\"min\":0,\"max\":4,\"step\":0.01},"
            "{\"key\":\"v8_wave\",\"name\":\"V8 Wave\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v8_penv\",\"name\":\"V8 P.Env\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v8_mix\",\"name\":\"V8 Mix\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v8_cutoff\",\"name\":\"V8 Cutoff\",\"type\":\"int\",\"min\":20,\"max\":18000,\"step\":1},"
            "{\"key\":\"v8_dist\",\"name\":\"V8 Distort\",\"type\":\"float\",\"min\":0,\"max\":50,\"step\":0.5},"
            "{\"key\":\"v8_preset\",\"name\":\"V8 Preset\",\"type\":\"int\",\"min\":0,\"max\":40,\"step\":1},"
            "{\"key\":\"v1_attack\",\"name\":\"V1 Attack\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.001},"
            "{\"key\":\"v1_prate\",\"name\":\"V1 P.Rate\",\"type\":\"float\",\"min\":0,\"max\":2,\"step\":0.01},"
            "{\"key\":\"v1_lamt\",\"name\":\"V1 LFO Amt\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v1_lrate\",\"name\":\"V1 LFO Rt\",\"type\":\"float\",\"min\":0.1,\"max\":80,\"step\":0.5},"
            "{\"key\":\"v1_ftype\",\"name\":\"V1 F.Type\",\"type\":\"enum\",\"options\":[\"LP\",\"HP\",\"BP\"]},"
            "{\"key\":\"v1_fres\",\"name\":\"V1 Reso\",\"type\":\"float\",\"min\":1,\"max\":5,\"step\":0.1},"
            "{\"key\":\"v1_nattack\",\"name\":\"V1 N.Atk\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.001},"
            "{\"key\":\"v1_ndecay\",\"name\":\"V1 N.Dec\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v1_level\",\"name\":\"V1 Level\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v2_attack\",\"name\":\"V2 Attack\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.001},"
            "{\"key\":\"v2_prate\",\"name\":\"V2 P.Rate\",\"type\":\"float\",\"min\":0,\"max\":2,\"step\":0.01},"
            "{\"key\":\"v2_lamt\",\"name\":\"V2 LFO Amt\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v2_lrate\",\"name\":\"V2 LFO Rt\",\"type\":\"float\",\"min\":0.1,\"max\":80,\"step\":0.5},"
            "{\"key\":\"v2_ftype\",\"name\":\"V2 F.Type\",\"type\":\"enum\",\"options\":[\"LP\",\"HP\",\"BP\"]},"
            "{\"key\":\"v2_fres\",\"name\":\"V2 Reso\",\"type\":\"float\",\"min\":1,\"max\":5,\"step\":0.1},"
            "{\"key\":\"v2_nattack\",\"name\":\"V2 N.Atk\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.001},"
            "{\"key\":\"v2_ndecay\",\"name\":\"V2 N.Dec\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v2_level\",\"name\":\"V2 Level\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v3_attack\",\"name\":\"V3 Attack\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.001},"
            "{\"key\":\"v3_prate\",\"name\":\"V3 P.Rate\",\"type\":\"float\",\"min\":0,\"max\":2,\"step\":0.01},"
            "{\"key\":\"v3_lamt\",\"name\":\"V3 LFO Amt\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v3_lrate\",\"name\":\"V3 LFO Rt\",\"type\":\"float\",\"min\":0.1,\"max\":80,\"step\":0.5},"
            "{\"key\":\"v3_ftype\",\"name\":\"V3 F.Type\",\"type\":\"enum\",\"options\":[\"LP\",\"HP\",\"BP\"]},"
            "{\"key\":\"v3_fres\",\"name\":\"V3 Reso\",\"type\":\"float\",\"min\":1,\"max\":5,\"step\":0.1},"
            "{\"key\":\"v3_nattack\",\"name\":\"V3 N.Atk\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.001},"
            "{\"key\":\"v3_ndecay\",\"name\":\"V3 N.Dec\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v3_level\",\"name\":\"V3 Level\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v4_attack\",\"name\":\"V4 Attack\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.001},"
            "{\"key\":\"v4_prate\",\"name\":\"V4 P.Rate\",\"type\":\"float\",\"min\":0,\"max\":2,\"step\":0.01},"
            "{\"key\":\"v4_lamt\",\"name\":\"V4 LFO Amt\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v4_lrate\",\"name\":\"V4 LFO Rt\",\"type\":\"float\",\"min\":0.1,\"max\":80,\"step\":0.5},"
            "{\"key\":\"v4_ftype\",\"name\":\"V4 F.Type\",\"type\":\"enum\",\"options\":[\"LP\",\"HP\",\"BP\"]},"
            "{\"key\":\"v4_fres\",\"name\":\"V4 Reso\",\"type\":\"float\",\"min\":1,\"max\":5,\"step\":0.1},"
            "{\"key\":\"v4_nattack\",\"name\":\"V4 N.Atk\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.001},"
            "{\"key\":\"v4_ndecay\",\"name\":\"V4 N.Dec\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v4_level\",\"name\":\"V4 Level\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v5_attack\",\"name\":\"V5 Attack\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.001},"
            "{\"key\":\"v5_prate\",\"name\":\"V5 P.Rate\",\"type\":\"float\",\"min\":0,\"max\":2,\"step\":0.01},"
            "{\"key\":\"v5_lamt\",\"name\":\"V5 LFO Amt\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v5_lrate\",\"name\":\"V5 LFO Rt\",\"type\":\"float\",\"min\":0.1,\"max\":80,\"step\":0.5},"
            "{\"key\":\"v5_ftype\",\"name\":\"V5 F.Type\",\"type\":\"enum\",\"options\":[\"LP\",\"HP\",\"BP\"]},"
            "{\"key\":\"v5_fres\",\"name\":\"V5 Reso\",\"type\":\"float\",\"min\":1,\"max\":5,\"step\":0.1},"
            "{\"key\":\"v5_nattack\",\"name\":\"V5 N.Atk\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.001},"
            "{\"key\":\"v5_ndecay\",\"name\":\"V5 N.Dec\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v5_level\",\"name\":\"V5 Level\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v6_attack\",\"name\":\"V6 Attack\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.001},"
            "{\"key\":\"v6_prate\",\"name\":\"V6 P.Rate\",\"type\":\"float\",\"min\":0,\"max\":2,\"step\":0.01},"
            "{\"key\":\"v6_lamt\",\"name\":\"V6 LFO Amt\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v6_lrate\",\"name\":\"V6 LFO Rt\",\"type\":\"float\",\"min\":0.1,\"max\":80,\"step\":0.5},"
            "{\"key\":\"v6_ftype\",\"name\":\"V6 F.Type\",\"type\":\"enum\",\"options\":[\"LP\",\"HP\",\"BP\"]},"
            "{\"key\":\"v6_fres\",\"name\":\"V6 Reso\",\"type\":\"float\",\"min\":1,\"max\":5,\"step\":0.1},"
            "{\"key\":\"v6_nattack\",\"name\":\"V6 N.Atk\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.001},"
            "{\"key\":\"v6_ndecay\",\"name\":\"V6 N.Dec\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v6_level\",\"name\":\"V6 Level\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v7_attack\",\"name\":\"V7 Attack\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.001},"
            "{\"key\":\"v7_prate\",\"name\":\"V7 P.Rate\",\"type\":\"float\",\"min\":0,\"max\":2,\"step\":0.01},"
            "{\"key\":\"v7_lamt\",\"name\":\"V7 LFO Amt\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v7_lrate\",\"name\":\"V7 LFO Rt\",\"type\":\"float\",\"min\":0.1,\"max\":80,\"step\":0.5},"
            "{\"key\":\"v7_ftype\",\"name\":\"V7 F.Type\",\"type\":\"enum\",\"options\":[\"LP\",\"HP\",\"BP\"]},"
            "{\"key\":\"v7_fres\",\"name\":\"V7 Reso\",\"type\":\"float\",\"min\":1,\"max\":5,\"step\":0.1},"
            "{\"key\":\"v7_nattack\",\"name\":\"V7 N.Atk\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.001},"
            "{\"key\":\"v7_ndecay\",\"name\":\"V7 N.Dec\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v7_level\",\"name\":\"V7 Level\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v8_attack\",\"name\":\"V8 Attack\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.001},"
            "{\"key\":\"v8_prate\",\"name\":\"V8 P.Rate\",\"type\":\"float\",\"min\":0,\"max\":2,\"step\":0.01},"
            "{\"key\":\"v8_lamt\",\"name\":\"V8 LFO Amt\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v8_lrate\",\"name\":\"V8 LFO Rt\",\"type\":\"float\",\"min\":0.1,\"max\":80,\"step\":0.5},"
            "{\"key\":\"v8_ftype\",\"name\":\"V8 F.Type\",\"type\":\"enum\",\"options\":[\"LP\",\"HP\",\"BP\"]},"
            "{\"key\":\"v8_fres\",\"name\":\"V8 Reso\",\"type\":\"float\",\"min\":1,\"max\":5,\"step\":0.1},"
            "{\"key\":\"v8_nattack\",\"name\":\"V8 N.Atk\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.001},"
            "{\"key\":\"v8_ndecay\",\"name\":\"V8 N.Dec\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v8_level\",\"name\":\"V8 Level\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01}"
            "]";
        int len = (int)strlen(cp);
        if (len >= buf_len) return -1;
        memcpy(buf, cp, len + 1);
        return len;
    }

    /* ui_hierarchy — MUST be in get_param for synths (module.json alone not enough) */
    if (strcmp(key, "ui_hierarchy") == 0) {
        static const char *hier =
            "{\"modes\":null,\"levels\":{"
            "\"root\":{\"name\":\"Weird Drum\","
            "\"knobs\":[\"v1_vol\",\"v2_vol\",\"v3_vol\",\"v4_vol\",\"v5_vol\",\"v6_vol\",\"v7_vol\",\"v8_vol\"],"
            "\"params\":[{\"level\":\"Mixer\",\"label\":\"Mixer\"},{\"level\":\"General\",\"label\":\"General\"},{\"level\":\"Patch\",\"label\":\"Patch\"},{\"level\":\"Pan\",\"label\":\"Pan\"},"
            "{\"level\":\"Voice 1\",\"label\":\"Voice 1\"},{\"level\":\"Voice 2\",\"label\":\"Voice 2\"},"
            "{\"level\":\"Voice 3\",\"label\":\"Voice 3\"},{\"level\":\"Voice 4\",\"label\":\"Voice 4\"},"
            "{\"level\":\"Voice 5\",\"label\":\"Voice 5\"},{\"level\":\"Voice 6\",\"label\":\"Voice 6\"},"
            "{\"level\":\"Voice 7\",\"label\":\"Voice 7\"},{\"level\":\"Voice 8\",\"label\":\"Voice 8\"}]},"
            "\"Mixer\":{\"label\":\"Mixer\","
            "\"knobs\":[\"v1_vol\",\"v2_vol\",\"v3_vol\",\"v4_vol\",\"v5_vol\",\"v6_vol\",\"v7_vol\",\"v8_vol\"],"
            "\"params\":[\"v1_vol\",\"v2_vol\",\"v3_vol\",\"v4_vol\",\"v5_vol\",\"v6_vol\",\"v7_vol\",\"v8_vol\"]},"
            "\"General\":{\"label\":\"General\","
            "\"knobs\":[\"comp\",\"dj_filter\",\"eq_lo\",\"lo_freq\",\"eq_mid\",\"mid_freq\",\"eq_hi\",\"hi_freq\"],"
            "\"params\":[\"comp\",\"dj_filter\",\"eq_lo\",\"lo_freq\",\"eq_mid\",\"mid_freq\",\"eq_hi\",\"hi_freq\",\"q_lo\",\"q_mid\",\"q_hi\",\"master\"]},"
            "\"Patch\":{\"label\":\"Patch\","
            "\"knobs\":[\"kit\",\"rnd_voice\",\"rnd_patch\",\"same_freq\",\"rnd_pan\"],"
            "\"params\":[\"kit\",\"rnd_voice\",\"rnd_patch\",\"same_freq\",\"rnd_pan\"]},"
            "\"Pan\":{\"label\":\"Pan\","
            "\"knobs\":[\"v1_pan\",\"v2_pan\",\"v3_pan\",\"v4_pan\",\"v5_pan\",\"v6_pan\",\"v7_pan\",\"v8_pan\"],"
            "\"params\":[\"v1_pan\",\"v2_pan\",\"v3_pan\",\"v4_pan\",\"v5_pan\",\"v6_pan\",\"v7_pan\",\"v8_pan\"]},"
            "\"Voice 1\":{\"label\":\"Voice 1\","
            "\"knobs\":[\"v1_freq\",\"v1_decay\",\"v1_wave\",\"v1_penv\",\"v1_mix\",\"v1_cutoff\",\"v1_dist\",\"v1_preset\"],"
            "\"params\":[\"v1_freq\",\"v1_attack\",\"v1_decay\",\"v1_wave\",\"v1_penv\",\"v1_prate\",\"v1_lamt\",\"v1_lrate\",\"v1_ftype\",\"v1_cutoff\",\"v1_fres\",\"v1_nattack\",\"v1_ndecay\",\"v1_mix\",\"v1_dist\",\"v1_level\",\"v1_preset\"]},"
            "\"Voice 2\":{\"label\":\"Voice 2\","
            "\"knobs\":[\"v2_freq\",\"v2_decay\",\"v2_wave\",\"v2_penv\",\"v2_mix\",\"v2_cutoff\",\"v2_dist\",\"v2_preset\"],"
            "\"params\":[\"v2_freq\",\"v2_attack\",\"v2_decay\",\"v2_wave\",\"v2_penv\",\"v2_prate\",\"v2_lamt\",\"v2_lrate\",\"v2_ftype\",\"v2_cutoff\",\"v2_fres\",\"v2_nattack\",\"v2_ndecay\",\"v2_mix\",\"v2_dist\",\"v2_level\",\"v2_preset\"]},"
            "\"Voice 3\":{\"label\":\"Voice 3\","
            "\"knobs\":[\"v3_freq\",\"v3_decay\",\"v3_wave\",\"v3_penv\",\"v3_mix\",\"v3_cutoff\",\"v3_dist\",\"v3_preset\"],"
            "\"params\":[\"v3_freq\",\"v3_attack\",\"v3_decay\",\"v3_wave\",\"v3_penv\",\"v3_prate\",\"v3_lamt\",\"v3_lrate\",\"v3_ftype\",\"v3_cutoff\",\"v3_fres\",\"v3_nattack\",\"v3_ndecay\",\"v3_mix\",\"v3_dist\",\"v3_level\",\"v3_preset\"]},"
            "\"Voice 4\":{\"label\":\"Voice 4\","
            "\"knobs\":[\"v4_freq\",\"v4_decay\",\"v4_wave\",\"v4_penv\",\"v4_mix\",\"v4_cutoff\",\"v4_dist\",\"v4_preset\"],"
            "\"params\":[\"v4_freq\",\"v4_attack\",\"v4_decay\",\"v4_wave\",\"v4_penv\",\"v4_prate\",\"v4_lamt\",\"v4_lrate\",\"v4_ftype\",\"v4_cutoff\",\"v4_fres\",\"v4_nattack\",\"v4_ndecay\",\"v4_mix\",\"v4_dist\",\"v4_level\",\"v4_preset\"]},"
            "\"Voice 5\":{\"label\":\"Voice 5\","
            "\"knobs\":[\"v5_freq\",\"v5_decay\",\"v5_wave\",\"v5_penv\",\"v5_mix\",\"v5_cutoff\",\"v5_dist\",\"v5_preset\"],"
            "\"params\":[\"v5_freq\",\"v5_attack\",\"v5_decay\",\"v5_wave\",\"v5_penv\",\"v5_prate\",\"v5_lamt\",\"v5_lrate\",\"v5_ftype\",\"v5_cutoff\",\"v5_fres\",\"v5_nattack\",\"v5_ndecay\",\"v5_mix\",\"v5_dist\",\"v5_level\",\"v5_preset\"]},"
            "\"Voice 6\":{\"label\":\"Voice 6\","
            "\"knobs\":[\"v6_freq\",\"v6_decay\",\"v6_wave\",\"v6_penv\",\"v6_mix\",\"v6_cutoff\",\"v6_dist\",\"v6_preset\"],"
            "\"params\":[\"v6_freq\",\"v6_attack\",\"v6_decay\",\"v6_wave\",\"v6_penv\",\"v6_prate\",\"v6_lamt\",\"v6_lrate\",\"v6_ftype\",\"v6_cutoff\",\"v6_fres\",\"v6_nattack\",\"v6_ndecay\",\"v6_mix\",\"v6_dist\",\"v6_level\",\"v6_preset\"]},"
            "\"Voice 7\":{\"label\":\"Voice 7\","
            "\"knobs\":[\"v7_freq\",\"v7_decay\",\"v7_wave\",\"v7_penv\",\"v7_mix\",\"v7_cutoff\",\"v7_dist\",\"v7_preset\"],"
            "\"params\":[\"v7_freq\",\"v7_attack\",\"v7_decay\",\"v7_wave\",\"v7_penv\",\"v7_prate\",\"v7_lamt\",\"v7_lrate\",\"v7_ftype\",\"v7_cutoff\",\"v7_fres\",\"v7_nattack\",\"v7_ndecay\",\"v7_mix\",\"v7_dist\",\"v7_level\",\"v7_preset\"]},"
            "\"Voice 8\":{\"label\":\"Voice 8\","
            "\"knobs\":[\"v8_freq\",\"v8_decay\",\"v8_wave\",\"v8_penv\",\"v8_mix\",\"v8_cutoff\",\"v8_dist\",\"v8_preset\"],"
            "\"params\":[\"v8_freq\",\"v8_attack\",\"v8_decay\",\"v8_wave\",\"v8_penv\",\"v8_prate\",\"v8_lamt\",\"v8_lrate\",\"v8_ftype\",\"v8_cutoff\",\"v8_fres\",\"v8_nattack\",\"v8_ndecay\",\"v8_mix\",\"v8_dist\",\"v8_level\",\"v8_preset\"]}"
            "}}";
        int len = (int)strlen(hier);
        if (len >= buf_len) return -1;
        memcpy(buf, hier, len + 1);
        return len;
    }

    /* State serialization */
    if (strcmp(key, "state") == 0) {
        int n = 0;
        /* Mixer volumes */
        for (int i = 0; i < NUM_VOICES; i++)
            n += snprintf(buf + n, buf_len - n, "%.3f ", inst->voice_vol[i]);
        /* Master */
        n += snprintf(buf + n, buf_len - n, "%.3f %.4f %.1f %.1f %.1f %.0f %.0f %.0f %.2f %.2f %.2f %.3f ",
            inst->master.comp_amount, inst->master.dj_filter,
            inst->master.eq_low_gain, inst->master.eq_mid_gain, inst->master.eq_high_gain,
            inst->master.eq_low_freq, inst->master.eq_mid_freq, inst->master.eq_high_freq,
            inst->master.eq_low_q, inst->master.eq_mid_q, inst->master.eq_high_q,
            inst->master.master_level);
        /* Per-voice */
        for (int i = 0; i < NUM_VOICES; i++) {
            wd_voice_t *v = &inst->voice[i];
            n += snprintf(buf + n, buf_len - n,
                "%d %.1f %.4f %.4f %.4f %.3f %.4f %.3f %.1f %d %.0f %.2f %.4f %.4f %.3f %.1f %.3f %.2f ",
                v->preset, v->freq, v->attack, v->decay, v->wave,
                v->pitch_env_amt, v->pitch_env_rate, v->pitch_lfo_amt, v->pitch_lfo_rate,
                v->filter_type, v->filter_cutoff, v->filter_res,
                v->noise_attack, v->noise_decay, v->mix, v->distortion, v->level, v->pan);
        }
        if (n >= buf_len) n = buf_len - 1;
        return n;
    }

    /* Direct parameter reads — all params must be readable for menu editing */

    /* Mixer volumes */
    for (int i = 0; i < NUM_VOICES; i++) {
        char k[24];
        snprintf(k, sizeof(k), "v%d_vol", i+1);
        if (strcmp(key, k) == 0) return snprintf(buf, buf_len, "%.4f", inst->voice_vol[i]);
    }

    /* Master params */
    if (strcmp(key, "comp") == 0) return snprintf(buf, buf_len, "%.4f", inst->master.comp_amount);
    if (strcmp(key, "dj_filter") == 0) return snprintf(buf, buf_len, "%.4f", inst->master.dj_filter);
    if (strcmp(key, "eq_lo") == 0) return snprintf(buf, buf_len, "%.4f", inst->master.eq_low_gain);
    if (strcmp(key, "eq_mid") == 0) return snprintf(buf, buf_len, "%.4f", inst->master.eq_mid_gain);
    if (strcmp(key, "eq_hi") == 0) return snprintf(buf, buf_len, "%.4f", inst->master.eq_high_gain);
    if (strcmp(key, "lo_freq") == 0) return snprintf(buf, buf_len, "%d", (int)inst->master.eq_low_freq);
    if (strcmp(key, "mid_freq") == 0) return snprintf(buf, buf_len, "%d", (int)inst->master.eq_mid_freq);
    if (strcmp(key, "hi_freq") == 0) return snprintf(buf, buf_len, "%d", (int)inst->master.eq_high_freq);
    if (strcmp(key, "q_lo") == 0) return snprintf(buf, buf_len, "%.4f", inst->master.eq_low_q);
    if (strcmp(key, "q_mid") == 0) return snprintf(buf, buf_len, "%.4f", inst->master.eq_mid_q);
    if (strcmp(key, "q_hi") == 0) return snprintf(buf, buf_len, "%.4f", inst->master.eq_high_q);
    if (strcmp(key, "kit") == 0) return snprintf(buf, buf_len, "%d", inst->current_kit);
    if (strcmp(key, "rnd_voice") == 0) return snprintf(buf, buf_len, "0");
    if (strcmp(key, "rnd_patch") == 0) return snprintf(buf, buf_len, "0");
    if (strcmp(key, "rnd_pan") == 0) return snprintf(buf, buf_len, "0");
    if (strcmp(key, "same_freq") == 0) return snprintf(buf, buf_len, "%d", inst->same_freq > 0 ? (int)inst->same_freq : 0);
    if (strcmp(key, "master") == 0) return snprintf(buf, buf_len, "%.4f", inst->master.master_level);

    /* Per-voice params */
    for (int i = 0; i < NUM_VOICES; i++) {
        char k[24];
        wd_voice_t *v = &inst->voice[i];

        snprintf(k, sizeof(k), "v%d_freq", i+1);
        if (strcmp(key, k) == 0) return snprintf(buf, buf_len, "%d", (int)v->freq);
        snprintf(k, sizeof(k), "v%d_decay", i+1);
        if (strcmp(key, k) == 0) return snprintf(buf, buf_len, "%.4f", v->decay);
        snprintf(k, sizeof(k), "v%d_wave", i+1);
        if (strcmp(key, k) == 0) return snprintf(buf, buf_len, "%.4f", v->wave);
        snprintf(k, sizeof(k), "v%d_penv", i+1);
        if (strcmp(key, k) == 0) return snprintf(buf, buf_len, "%.4f", v->pitch_env_amt);
        snprintf(k, sizeof(k), "v%d_mix", i+1);
        if (strcmp(key, k) == 0) return snprintf(buf, buf_len, "%.4f", v->mix);
        snprintf(k, sizeof(k), "v%d_cutoff", i+1);
        if (strcmp(key, k) == 0) return snprintf(buf, buf_len, "%d", (int)v->filter_cutoff);
        snprintf(k, sizeof(k), "v%d_dist", i+1);
        if (strcmp(key, k) == 0) return snprintf(buf, buf_len, "%.4f", v->distortion);
        snprintf(k, sizeof(k), "v%d_preset", i+1);
        if (strcmp(key, k) == 0) return snprintf(buf, buf_len, "%d", v->preset);
        /* Additional voice params (menu-only, not on knobs) */
        snprintf(k, sizeof(k), "v%d_attack", i+1);
        if (strcmp(key, k) == 0) return snprintf(buf, buf_len, "%.4f", v->attack);
        snprintf(k, sizeof(k), "v%d_prate", i+1);
        if (strcmp(key, k) == 0) return snprintf(buf, buf_len, "%.4f", v->pitch_env_rate);
        snprintf(k, sizeof(k), "v%d_lamt", i+1);
        if (strcmp(key, k) == 0) return snprintf(buf, buf_len, "%.4f", v->pitch_lfo_amt);
        snprintf(k, sizeof(k), "v%d_lrate", i+1);
        if (strcmp(key, k) == 0) return snprintf(buf, buf_len, "%.4f", v->pitch_lfo_rate);
        snprintf(k, sizeof(k), "v%d_ftype", i+1);
        if (strcmp(key, k) == 0) return snprintf(buf, buf_len, "%s", FILTER_NAMES[v->filter_type]);
        snprintf(k, sizeof(k), "v%d_fres", i+1);
        if (strcmp(key, k) == 0) return snprintf(buf, buf_len, "%.4f", v->filter_res);
        snprintf(k, sizeof(k), "v%d_nattack", i+1);
        if (strcmp(key, k) == 0) return snprintf(buf, buf_len, "%.4f", v->noise_attack);
        snprintf(k, sizeof(k), "v%d_ndecay", i+1);
        if (strcmp(key, k) == 0) return snprintf(buf, buf_len, "%.4f", v->noise_decay);
        snprintf(k, sizeof(k), "v%d_level", i+1);
        if (strcmp(key, k) == 0) return snprintf(buf, buf_len, "%.4f", v->level);
        snprintf(k, sizeof(k), "v%d_pan", i+1);
        if (strcmp(key, k) == 0) return snprintf(buf, buf_len, "%.4f", v->pan);
    }

    return -1;
}

/* ── render_block ── */
static void render_block(void *instance, int16_t *out_lr, int frames) {
    wd_instance_t *inst = (wd_instance_t *)instance;
    if (!inst) {
        memset(out_lr, 0, frames * 2 * sizeof(int16_t));
        return;
    }

    for (int i = 0; i < frames; i++) {
        float mix_l = 0.0f, mix_r = 0.0f;

        for (int v = 0; v < NUM_VOICES; v++) {
            float vol = onepole(&inst->voice_vol_smooth[v], inst->voice_vol[v], 0.002f);
            float pan = onepole(&inst->voice_pan_smooth[v], inst->voice[v].pan, 0.002f);
            float sample = voice_render_sample(&inst->voice[v]) * vol;
            /* Constant-power panning: L = cos(angle), R = sin(angle) */
            float angle = (pan + 1.0f) * 0.25f * 3.14159265f; /* 0..pi/2 */
            mix_l += sample * cosf(angle);
            mix_r += sample * sinf(angle);
        }

        /* Scale down (8 voices) */
        mix_l *= 0.35f;
        mix_r *= 0.35f;

        /* Master FX (process L and R independently) */
        /* Use a single mono master for simplicity — average then re-spread */
        float mono = (mix_l + mix_r) * 0.5f;
        float stereo_diff = mix_l - mix_r;
        mono = master_process(&inst->master, mono);
        float out_l = mono + stereo_diff * 0.5f;
        float out_r = mono - stereo_diff * 0.5f;

        /* Clamp and output stereo int16 */
        out_l = clampf(out_l, -1.0f, 1.0f);
        out_r = clampf(out_r, -1.0f, 1.0f);
        out_lr[i * 2]     = (int16_t)(out_l * 32767.0f);
        out_lr[i * 2 + 1] = (int16_t)(out_r * 32767.0f);
    }
}

/* ============================================================================
 * Plugin API Export — 8 fields (get_error between get_param and render_block)
 * ============================================================================ */

typedef struct {
    uint32_t api_version;
    void* (*create_instance)(const char *, const char *);
    void  (*destroy_instance)(void *);
    void  (*on_midi)(void *, const uint8_t *, int, int);
    void  (*set_param)(void *, const char *, const char *);
    int   (*get_param)(void *, const char *, char *, int);
    int   (*get_error)(void *, char *, int);     /* MUST exist — NULL is fine */
    void  (*render_block)(void *, int16_t *, int);
} plugin_api_v2_t;

__attribute__((visibility("default")))
plugin_api_v2_t* move_plugin_init_v2(const void *host) {
    (void)host;
    static plugin_api_v2_t api = {
        .api_version      = 2,
        .create_instance  = create_instance,
        .destroy_instance = destroy_instance,
        .on_midi          = on_midi,
        .set_param        = set_param,
        .get_param        = get_param,
        .get_error        = NULL,
        .render_block     = render_block,
    };
    return &api;
}
