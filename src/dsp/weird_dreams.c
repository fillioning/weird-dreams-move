/* ============================================================================
 * Weird Dreams — 8-voice analog drum synthesizer for Ableton Move
 * Based on WeirdDrums by Daniele Filaretti (MIT)
 * https://github.com/dfilaretti/WeirdDrums
 * Ported and expanded for Schwung by Vincent Fillion
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
 * Master bus: compressor, distortion, 3-band EQ, reverb, delay
 * UI: Patch page, General page, dynamic Voice page, FX page
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
    float reverb_send;      /* 0..1 */
    float delay_send;       /* 0..1 */

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
    v->reverb_send = 0.0f; v->delay_send = 0.0f;
}

static void voice_apply_preset(wd_voice_t *v, int preset) {
    v->preset = preset;
    preset_defaults(v);
    switch (preset) {
    /* ── KICKS (0-4) ── */
    case 0: /* Sub Kick — deep 808 sub */
        v->freq=38; v->wave=0.0f; v->decay=0.8f; v->pitch_env_amt=0.95f; v->pitch_env_rate=0.07f;
        v->filter_type=0; v->filter_cutoff=250; v->filter_res=1.2f;
        v->noise_decay=0.02f; v->mix=0.02f; v->distortion=5; v->level=0.95f;
        v->reverb_send=0.05f; v->delay_send=0.0f; break;
    case 1: /* Punch Kick — tight 909 */
        v->freq=65; v->wave=0.0f; v->attack=0.0003f; v->decay=0.15f; v->pitch_env_amt=0.75f; v->pitch_env_rate=0.02f;
        v->filter_type=0; v->filter_cutoff=600; v->filter_res=1.5f;
        v->noise_decay=0.015f; v->mix=0.1f; v->distortion=14; v->level=0.9f;
        v->reverb_send=0.07f; v->delay_send=0.0f; break;
    case 2: /* FM Kick — saw sweep, Microtonic */
        v->freq=48; v->wave=0.66f; v->attack=0.0001f; v->decay=0.35f; v->pitch_env_amt=1.0f; v->pitch_env_rate=0.04f;
        v->pitch_lfo_amt=0.03f; v->pitch_lfo_rate=8;
        v->filter_type=0; v->filter_cutoff=800; v->filter_res=2.0f;
        v->noise_decay=0.04f; v->mix=0.08f; v->distortion=20; v->level=0.85f;
        v->reverb_send=0.08f; v->delay_send=0.0f; break;
    case 3: /* Dusty Kick — lo-fi saturated */
        v->freq=50; v->wave=0.15f; v->decay=0.2f; v->pitch_env_amt=0.6f; v->pitch_env_rate=0.025f;
        v->filter_type=0; v->filter_cutoff=500; v->filter_res=1.3f;
        v->noise_decay=0.06f; v->mix=0.12f; v->distortion=25; v->level=0.85f;
        v->reverb_send=0.1f; v->delay_send=0.0f; break;
    case 4: /* Long Kick — boomy tail */
        v->freq=42; v->wave=0.0f; v->decay=1.2f; v->pitch_env_amt=0.85f; v->pitch_env_rate=0.08f;
        v->filter_type=0; v->filter_cutoff=350; v->filter_res=1.0f;
        v->noise_decay=0.03f; v->mix=0.03f; v->distortion=3; v->level=0.9f;
        v->reverb_send=0.06f; v->delay_send=0.0f; break;

    /* ── SNARES (5-9) ── */
    case 5: /* Snare — classic analog */
        v->freq=190; v->wave=0.0f; v->attack=0.0005f; v->decay=0.16f; v->pitch_env_amt=0.35f; v->pitch_env_rate=0.025f;
        v->filter_type=2; v->filter_cutoff=3500; v->filter_res=1.4f;
        v->noise_decay=0.16f; v->mix=0.6f; v->distortion=4; v->level=0.8f;
        v->reverb_send=0.2f; v->delay_send=0.05f; break;
    case 6: /* Snap Snare — crispy 909 */
        v->freq=220; v->wave=0.0f; v->attack=0.0003f; v->decay=0.1f; v->pitch_env_amt=0.4f; v->pitch_env_rate=0.015f;
        v->filter_type=1; v->filter_cutoff=4000; v->filter_res=1.8f;
        v->noise_attack=0.0003f; v->noise_decay=0.12f; v->mix=0.7f; v->distortion=6; v->level=0.75f;
        v->reverb_send=0.15f; v->delay_send=0.07f; break;
    case 7: /* Fat Snare — thick body */
        v->freq=150; v->wave=0.15f; v->decay=0.25f; v->pitch_env_amt=0.2f; v->pitch_env_rate=0.04f;
        v->filter_type=0; v->filter_cutoff=2000; v->filter_res=1.0f;
        v->noise_decay=0.2f; v->mix=0.45f; v->distortion=5; v->level=0.85f;
        v->reverb_send=0.25f; v->delay_send=0.08f; break;
    case 8: /* Crack Snare — harsh transient */
        v->freq=350; v->wave=1.0f; v->attack=0.0001f; v->decay=0.06f; v->pitch_env_amt=0.5f; v->pitch_env_rate=0.008f;
        v->filter_type=2; v->filter_cutoff=5000; v->filter_res=3.5f;
        v->noise_attack=0.0001f; v->noise_decay=0.08f; v->mix=0.75f; v->distortion=16; v->level=0.7f;
        v->reverb_send=0.3f; v->delay_send=0.1f; break;
    case 9: /* Noise Snare — pure noise */
        v->freq=300; v->wave=1.0f; v->attack=0.0001f; v->decay=0.12f; v->pitch_env_amt=0.0f; v->pitch_env_rate=0.01f;
        v->filter_type=2; v->filter_cutoff=4500; v->filter_res=1.2f;
        v->noise_attack=0.0001f; v->noise_decay=0.15f; v->mix=0.95f; v->distortion=2; v->level=0.7f;
        v->reverb_send=0.18f; v->delay_send=0.06f; break;

    /* ── TOMS (10-14) ── */
    case 10: /* Low Tom */
        v->freq=75; v->wave=0.0f; v->decay=0.4f; v->pitch_env_amt=0.55f; v->pitch_env_rate=0.045f;
        v->filter_type=0; v->filter_cutoff=700; v->filter_res=1.0f;
        v->noise_decay=0.08f; v->mix=0.1f; v->distortion=3; v->level=0.8f;
        v->reverb_send=0.25f; v->delay_send=0.0f; break;
    case 11: /* Mid Tom */
        v->freq=130; v->wave=0.0f; v->decay=0.3f; v->pitch_env_amt=0.45f; v->pitch_env_rate=0.035f;
        v->filter_type=0; v->filter_cutoff=900; v->filter_res=1.0f;
        v->noise_decay=0.06f; v->mix=0.08f; v->distortion=2; v->level=0.8f;
        v->reverb_send=0.22f; v->delay_send=0.03f; break;
    case 12: /* High Tom */
        v->freq=200; v->wave=0.0f; v->decay=0.22f; v->pitch_env_amt=0.4f; v->pitch_env_rate=0.03f;
        v->filter_type=0; v->filter_cutoff=1200; v->filter_res=1.0f;
        v->noise_decay=0.05f; v->mix=0.08f; v->distortion=2; v->level=0.75f;
        v->reverb_send=0.2f; v->delay_send=0.04f; break;
    case 13: /* Acid Tom — resonant filter sweep */
        v->freq=110; v->wave=0.66f; v->decay=0.35f; v->pitch_env_amt=0.7f; v->pitch_env_rate=0.06f;
        v->filter_type=0; v->filter_cutoff=1800; v->filter_res=3.8f;
        v->noise_decay=0.05f; v->mix=0.05f; v->distortion=10; v->level=0.75f;
        v->reverb_send=0.28f; v->delay_send=0.05f; break;
    case 14: /* Conga — long resonant body */
        v->freq=240; v->wave=0.0f; v->attack=0.0005f; v->decay=0.5f; v->pitch_env_amt=0.25f; v->pitch_env_rate=0.02f;
        v->filter_type=2; v->filter_cutoff=1200; v->filter_res=2.5f;
        v->noise_attack=0.0005f; v->noise_decay=0.06f; v->mix=0.15f; v->distortion=6; v->level=0.7f;
        v->reverb_send=0.3f; v->delay_send=0.02f; break;

    /* ── HI-HATS (15-19) ── */
    case 15: /* Closed HH — tight */
        v->freq=450; v->wave=1.0f; v->attack=0.0001f; v->decay=0.035f; v->pitch_env_amt=0.0f; v->pitch_env_rate=0.01f;
        v->pitch_lfo_amt=0.04f; v->pitch_lfo_rate=60;
        v->filter_type=1; v->filter_cutoff=9000; v->filter_res=2.0f;
        v->noise_attack=0.0001f; v->noise_decay=0.035f; v->mix=0.92f; v->distortion=3; v->level=0.55f;
        v->reverb_send=0.1f; v->delay_send=0.0f; break;
    case 16: /* Open HH — sizzle */
        v->freq=420; v->wave=1.0f; v->attack=0.0001f; v->decay=0.35f; v->pitch_env_amt=0.0f; v->pitch_env_rate=0.01f;
        v->pitch_lfo_amt=0.06f; v->pitch_lfo_rate=40;
        v->filter_type=1; v->filter_cutoff=6500; v->filter_res=1.8f;
        v->noise_attack=0.0001f; v->noise_decay=0.35f; v->mix=0.93f; v->distortion=2; v->level=0.5f;
        v->reverb_send=0.15f; v->delay_send=0.05f; break;
    case 17: /* Pedal HH — medium */
        v->freq=380; v->wave=1.0f; v->attack=0.0001f; v->decay=0.1f; v->pitch_env_amt=0.0f; v->pitch_env_rate=0.01f;
        v->pitch_lfo_amt=0.03f; v->pitch_lfo_rate=50;
        v->filter_type=1; v->filter_cutoff=7000; v->filter_res=1.5f;
        v->noise_attack=0.0001f; v->noise_decay=0.1f; v->mix=0.9f; v->distortion=1; v->level=0.5f;
        v->reverb_send=0.12f; v->delay_send=0.02f; break;
    case 18: /* Metallic HH — LXR-02 */
        v->freq=500; v->wave=1.0f; v->attack=0.0001f; v->decay=0.05f; v->pitch_env_amt=0.05f; v->pitch_env_rate=0.005f;
        v->pitch_lfo_amt=0.1f; v->pitch_lfo_rate=75;
        v->filter_type=2; v->filter_cutoff=8000; v->filter_res=3.0f;
        v->noise_attack=0.0001f; v->noise_decay=0.05f; v->mix=0.85f; v->distortion=8; v->level=0.55f;
        v->reverb_send=0.13f; v->delay_send=0.03f; break;
    case 19: /* Trashy HH — dirty, distorted */
        v->freq=550; v->wave=0.8f; v->attack=0.0001f; v->decay=0.08f; v->pitch_env_amt=0.0f; v->pitch_env_rate=0.01f;
        v->pitch_lfo_amt=0.15f; v->pitch_lfo_rate=65;
        v->filter_type=1; v->filter_cutoff=5000; v->filter_res=2.5f;
        v->noise_attack=0.0001f; v->noise_decay=0.08f; v->mix=0.88f; v->distortion=18; v->level=0.5f;
        v->reverb_send=0.11f; v->delay_send=0.04f; break;

    /* ── CYMBALS (20-24) ── */
    case 20: /* Crash — bright */
        v->freq=280; v->wave=1.0f; v->attack=0.002f; v->decay=1.2f; v->pitch_env_amt=0.0f; v->pitch_env_rate=0.01f;
        v->pitch_lfo_amt=0.07f; v->pitch_lfo_rate=4;
        v->filter_type=1; v->filter_cutoff=4500; v->filter_res=1.2f;
        v->noise_attack=0.002f; v->noise_decay=1.2f; v->mix=0.88f; v->distortion=0; v->level=0.45f;
        v->reverb_send=0.35f; v->delay_send=0.0f; break;
    case 21: /* Ride — bell character */
        v->freq=600; v->wave=1.0f; v->attack=0.0001f; v->decay=0.8f; v->pitch_env_amt=0.04f; v->pitch_env_rate=0.005f;
        v->pitch_lfo_amt=0.03f; v->pitch_lfo_rate=6;
        v->filter_type=2; v->filter_cutoff=5500; v->filter_res=2.0f;
        v->noise_attack=0.0001f; v->noise_decay=0.6f; v->mix=0.7f; v->distortion=4; v->level=0.5f;
        v->reverb_send=0.3f; v->delay_send=0.0f; break;
    case 22: /* Ride Bell — ping */
        v->freq=900; v->wave=1.0f; v->attack=0.0001f; v->decay=0.5f; v->pitch_env_amt=0.08f; v->pitch_env_rate=0.004f;
        v->filter_type=2; v->filter_cutoff=4000; v->filter_res=3.5f;
        v->noise_attack=0.0001f; v->noise_decay=0.2f; v->mix=0.4f; v->distortion=8; v->level=0.55f;
        v->reverb_send=0.25f; v->delay_send=0.0f; break;
    case 23: /* Dark Crash — filtered */
        v->freq=250; v->wave=0.85f; v->attack=0.003f; v->decay=1.5f; v->pitch_env_amt=0.0f; v->pitch_env_rate=0.01f;
        v->pitch_lfo_amt=0.05f; v->pitch_lfo_rate=3;
        v->filter_type=0; v->filter_cutoff=3000; v->filter_res=1.0f;
        v->noise_attack=0.003f; v->noise_decay=1.5f; v->mix=0.85f; v->distortion=0; v->level=0.45f;
        v->reverb_send=0.4f; v->delay_send=0.0f; break;
    case 24: /* Sizzle — long shimmer */
        v->freq=350; v->wave=1.0f; v->attack=0.005f; v->decay=2.5f; v->pitch_env_amt=0.0f; v->pitch_env_rate=0.01f;
        v->pitch_lfo_amt=0.1f; v->pitch_lfo_rate=5;
        v->filter_type=1; v->filter_cutoff=6000; v->filter_res=1.5f;
        v->noise_attack=0.005f; v->noise_decay=2.5f; v->mix=0.9f; v->distortion=0; v->level=0.4f;
        v->reverb_send=0.35f; v->delay_send=0.0f; break;

    /* ── CLAPS (25-29) — use clap retrigger for flutter ── */
    case 25: /* Clap 808 — classic triple-hit flutter */
        v->freq=800; v->wave=1.0f; v->attack=0.0001f; v->decay=0.2f; v->pitch_env_amt=0.0f; v->pitch_env_rate=0.01f;
        v->filter_type=2; v->filter_cutoff=1800; v->filter_res=1.5f;
        v->noise_attack=0.0001f; v->noise_decay=0.2f; v->mix=0.95f; v->distortion=4; v->level=0.7f;
        v->clap_count=3; v->clap_spacing=480;
        v->reverb_send=0.4f; v->delay_send=0.15f; break; /* 3 hits, ~11ms apart */
    case 26: /* Tight Clap — fast flutter, short tail */
        v->freq=1000; v->wave=1.0f; v->attack=0.0001f; v->decay=0.1f; v->pitch_env_amt=0.0f; v->pitch_env_rate=0.01f;
        v->filter_type=2; v->filter_cutoff=2500; v->filter_res=1.8f;
        v->noise_attack=0.0001f; v->noise_decay=0.1f; v->mix=0.92f; v->distortion=6; v->level=0.7f;
        v->clap_count=2; v->clap_spacing=350;
        v->reverb_send=0.3f; v->delay_send=0.1f; break;
    case 27: /* Big Clap — wide flutter, long reverby tail */
        v->freq=700; v->wave=1.0f; v->attack=0.0001f; v->decay=0.4f; v->pitch_env_amt=0.0f; v->pitch_env_rate=0.01f;
        v->filter_type=2; v->filter_cutoff=1500; v->filter_res=1.2f;
        v->noise_attack=0.0001f; v->noise_decay=0.4f; v->mix=0.93f; v->distortion=3; v->level=0.65f;
        v->clap_count=4; v->clap_spacing=550;
        v->reverb_send=0.5f; v->delay_send=0.2f; break; /* 4 hits, wider spacing */
    case 28: /* Dirty Clap — distorted */
        v->freq=900; v->wave=0.85f; v->attack=0.0001f; v->decay=0.15f; v->pitch_env_amt=0.0f; v->pitch_env_rate=0.01f;
        v->filter_type=2; v->filter_cutoff=2000; v->filter_res=2.0f;
        v->noise_attack=0.0001f; v->noise_decay=0.15f; v->mix=0.9f; v->distortion=18; v->level=0.65f;
        v->clap_count=3; v->clap_spacing=400;
        v->reverb_send=0.35f; v->delay_send=0.12f; break;
    case 29: /* Snap Clap — single hit, no flutter */
        v->freq=1200; v->wave=1.0f; v->attack=0.0001f; v->decay=0.08f; v->pitch_env_amt=0.1f; v->pitch_env_rate=0.005f;
        v->filter_type=1; v->filter_cutoff=3000; v->filter_res=2.5f;
        v->noise_attack=0.0001f; v->noise_decay=0.08f; v->mix=0.85f; v->distortion=8; v->level=0.7f;
        v->reverb_send=0.35f; v->delay_send=0.15f; break;

    /* ── PERCUSSION (30-34) ── */
    case 30: /* Rimshot — sharp click */
        v->freq=500; v->wave=1.0f; v->attack=0.0001f; v->decay=0.035f; v->pitch_env_amt=0.25f; v->pitch_env_rate=0.008f;
        v->filter_type=2; v->filter_cutoff=4000; v->filter_res=2.5f;
        v->noise_attack=0.0001f; v->noise_decay=0.03f; v->mix=0.4f; v->distortion=8; v->level=0.75f;
        v->reverb_send=0.1f; v->delay_send=0.05f; break;
    case 31: /* Cowbell — metallic ring */
        v->freq=560; v->wave=1.0f; v->attack=0.0001f; v->decay=0.12f; v->pitch_env_amt=0.06f; v->pitch_env_rate=0.004f;
        v->filter_type=2; v->filter_cutoff=2200; v->filter_res=3.5f;
        v->noise_attack=0.0001f; v->noise_decay=0.03f; v->mix=0.12f; v->distortion=14; v->level=0.65f;
        v->reverb_send=0.15f; v->delay_send=0.1f; break;
    case 32: /* Clave — woody click */
        v->freq=2500; v->wave=0.0f; v->attack=0.0001f; v->decay=0.02f; v->pitch_env_amt=0.15f; v->pitch_env_rate=0.005f;
        v->filter_type=2; v->filter_cutoff=3000; v->filter_res=3.0f;
        v->noise_attack=0.0001f; v->noise_decay=0.01f; v->mix=0.1f; v->distortion=8; v->level=0.7f;
        v->reverb_send=0.12f; v->delay_send=0.08f; break;
    case 33: /* Shaker — maracas */
        v->freq=600; v->wave=1.0f; v->attack=0.002f; v->decay=0.12f; v->pitch_env_amt=0.0f; v->pitch_env_rate=0.01f;
        v->filter_type=1; v->filter_cutoff=5000; v->filter_res=1.0f;
        v->noise_attack=0.002f; v->noise_decay=0.12f; v->mix=0.85f; v->distortion=0; v->level=0.5f;
        v->reverb_send=0.18f; v->delay_send=0.15f; break;
    case 34: /* Tamb — tambourine */
        v->freq=700; v->wave=1.0f; v->attack=0.0001f; v->decay=0.2f; v->pitch_env_amt=0.0f; v->pitch_env_rate=0.01f;
        v->pitch_lfo_amt=0.08f; v->pitch_lfo_rate=50;
        v->filter_type=1; v->filter_cutoff=7000; v->filter_res=2.0f;
        v->noise_attack=0.0001f; v->noise_decay=0.2f; v->mix=0.9f; v->distortion=2; v->level=0.5f;
        v->reverb_send=0.2f; v->delay_send=0.12f; break;

    /* ── FX (35-39) ── */
    case 35: /* Zap — laser sweep */
        v->freq=1200; v->wave=0.66f; v->attack=0.0001f; v->decay=0.1f; v->pitch_env_amt=1.0f; v->pitch_env_rate=0.1f;
        v->filter_type=0; v->filter_cutoff=8000; v->filter_res=2.5f;
        v->noise_attack=0.0001f; v->noise_decay=0.01f; v->mix=0.03f; v->distortion=18; v->level=0.7f;
        v->reverb_send=0.2f; v->delay_send=0.25f; break;
    case 36: /* Glitch — fast LFO chaos */
        v->freq=320; v->wave=1.0f; v->attack=0.0001f; v->decay=0.07f; v->pitch_env_amt=0.4f; v->pitch_env_rate=0.015f;
        v->pitch_lfo_amt=0.3f; v->pitch_lfo_rate=70;
        v->filter_type=2; v->filter_cutoff=3000; v->filter_res=3.0f;
        v->noise_attack=0.0001f; v->noise_decay=0.07f; v->mix=0.5f; v->distortion=22; v->level=0.6f;
        v->reverb_send=0.3f; v->delay_send=0.2f; break;
    case 37: /* Drone Hit — long throb */
        v->freq=60; v->wave=0.66f; v->attack=0.01f; v->decay=2.0f; v->pitch_env_amt=0.15f; v->pitch_env_rate=0.15f;
        v->pitch_lfo_amt=0.12f; v->pitch_lfo_rate=3.5f;
        v->filter_type=0; v->filter_cutoff=1500; v->filter_res=3.0f;
        v->noise_attack=0.01f; v->noise_decay=0.5f; v->mix=0.25f; v->distortion=8; v->level=0.65f;
        v->reverb_send=0.4f; v->delay_send=0.3f; break;
    case 38: /* Blip — short tonal ping */
        v->freq=1800; v->wave=0.0f; v->attack=0.0001f; v->decay=0.025f; v->pitch_env_amt=0.6f; v->pitch_env_rate=0.008f;
        v->filter_type=2; v->filter_cutoff=5000; v->filter_res=3.5f;
        v->noise_attack=0.0001f; v->noise_decay=0.01f; v->mix=0.05f; v->distortion=10; v->level=0.6f;
        v->reverb_send=0.25f; v->delay_send=0.15f; break;
    case 39: /* Noise Burst — white noise hit */
        v->freq=200; v->wave=1.0f; v->attack=0.0001f; v->decay=0.06f; v->pitch_env_amt=0.0f; v->pitch_env_rate=0.01f;
        v->filter_type=2; v->filter_cutoff=4000; v->filter_res=1.0f;
        v->noise_attack=0.0001f; v->noise_decay=0.06f; v->mix=1.0f; v->distortion=2; v->level=0.6f;
        v->reverb_send=0.35f; v->delay_send=0.2f; break;

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
        float gain = 1.0f;
        if (m->comp_env > threshold) {
            float over = m->comp_env - threshold;
            float target_over = over / ratio;
            gain = (threshold + target_over) / (threshold + over);
            out *= gain;
        }

        /* Volume-loss compensation: restore only what compression removed */
        if (gain < 1.0f) {
            float compensation = 1.0f / fmaxf(gain, 0.1f);  /* invert gain reduction */
            compensation = fminf(compensation, 3.0f);         /* cap at +9.5 dB */
            float blend = 0.5f * m->comp_amount;              /* partial restore: 0-50% */
            out *= 1.0f + (compensation - 1.0f) * blend;
        }

        /* Above 50%: add saturation that gets progressively filthier */
        if (m->comp_amount > 0.5f) {
            float dirt = (m->comp_amount - 0.5f) * 2.0f;  /* 0..1 over 50-100% */
            float drive = 1.0f + dirt * dirt * 4.0f;       /* 1x → 5x, quadratic */
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

/* ── Reverb (Schroeder network: 4 comb + 2 allpass) ── */
#define REV_MAX_COMB    2048
#define REV_MAX_AP      512

typedef struct {
    float buf[REV_MAX_COMB];
    int   len;
    int   idx;
    float feedback;
    float damp;         /* one-pole LPF state in feedback */
    float damp_state;
} wd_comb_t;

static float comb_process(wd_comb_t *c, float in) {
    int len = c->len;
    if (len < 1) return in;
    if (c->idx >= len) c->idx = 0;
    float out = c->buf[c->idx];
    c->damp_state = out * (1.0f - c->damp) + c->damp_state * c->damp;
    c->buf[c->idx] = in + c->damp_state * c->feedback;
    c->idx = (c->idx + 1) % len;
    return out;
}

typedef struct {
    float buf[REV_MAX_AP];
    int   len;
    int   idx;
    float feedback;
} wd_allpass_t;

static float allpass_process(wd_allpass_t *a, float in) {
    int len = a->len;
    if (len < 1) return in;
    if (a->idx >= len) a->idx = 0;
    float buf_out = a->buf[a->idx];
    float out = buf_out - in;
    a->buf[a->idx] = in + buf_out * a->feedback;
    a->idx = (a->idx + 1) % len;
    return out;
}

/* Reverb types: 0=Club, 1=Garage, 2=Studio */
#define REV_NUM_TYPES 3
static const char *REV_TYPE_NAMES[REV_NUM_TYPES] = { "Club", "Garage", "Studio" };

typedef struct {
    wd_comb_t    comb[8];    /* 0-3 = left, 4-7 = right */
    wd_allpass_t ap[4];      /* 0-1 = left, 2-3 = right */
    float        mix;        /* 0..1 dry/wet */
    int          type;       /* 0=Club, 1=Garage, 2=Studio */
    float        size;       /* 0..1 room size */
    float        decay;      /* 0..1 decay time */
    float        input_lpf;  /* pre-filter state */
} wd_reverb_t;

/* Comb delay lengths per type (in samples at 44100Hz) — prime-ish for diffusion
 * 8 combs: first 4 for left channel, next 4 for right (different primes for decorrelation) */
static const int COMB_LENS[REV_NUM_TYPES][8] = {
    { 557, 709, 877, 1013,   601, 743, 911, 1049 },    /* Club */
    { 1117, 1277, 1499, 1657, 1153, 1319, 1531, 1693 }, /* Garage */
    { 811, 991, 1151, 1327,  853, 1031, 1187, 1361 }    /* Studio */
};
static const int AP_LENS[REV_NUM_TYPES][4] = {
    { 131, 199, 139, 211 },   /* Club */
    { 211, 307, 227, 317 },   /* Garage */
    { 173, 257, 181, 269 }    /* Studio */
};
/* Damping per type: Club=bright reflective, Garage=mid metallic, Studio=warm smooth */
static const float TYPE_DAMP[REV_NUM_TYPES] = { 0.2f, 0.4f, 0.65f };
/* Feedback base per type: Club=short, Garage=long, Studio=medium-long */
static const float TYPE_FB_BASE[REV_NUM_TYPES] = { 0.7f, 0.82f, 0.78f };

static void reverb_init(wd_reverb_t *r) {
    memset(r, 0, sizeof(*r));
    r->mix = 0.0f;
    r->type = 0;
    r->size = 0.5f;
    r->decay = 0.5f;
}

static void reverb_set_type(wd_reverb_t *r, int type) {
    if (type < 0 || type >= REV_NUM_TYPES) type = 0;
    r->type = type;
    for (int i = 0; i < 8; i++) {
        int len = (int)(COMB_LENS[type][i] * (0.5f + r->size * 0.8f));
        if (len < 1) len = 1;
        if (len > REV_MAX_COMB) len = REV_MAX_COMB;
        r->comb[i].idx = 0;
        r->comb[i].len = len;
        r->comb[i].damp_state = 0.0f;
    }
    for (int i = 0; i < 4; i++) {
        int len = (int)(AP_LENS[type][i] * (0.5f + r->size * 0.8f));
        if (len < 1) len = 1;
        if (len > REV_MAX_AP) len = REV_MAX_AP;
        r->ap[i].idx = 0;
        r->ap[i].len = len;
        r->ap[i].feedback = 0.5f;
    }
}

static void reverb_update_params(wd_reverb_t *r) {
    int type = r->type;
    float fb = TYPE_FB_BASE[type] + r->decay * (0.98f - TYPE_FB_BASE[type]);
    float damp = TYPE_DAMP[type] * (1.0f - r->decay * 0.3f);
    for (int i = 0; i < 8; i++) {
        r->comb[i].feedback = fb;
        r->comb[i].damp = damp;
    }
}

static void reverb_process_stereo(wd_reverb_t *r, float in, float *out_l, float *out_r) {
    /* Pre-filter: gentle LPF to tame harshness */
    r->input_lpf += 0.4f * (in - r->input_lpf);
    float filtered = r->input_lpf;

    /* Left channel: combs 0-3 + allpass 0-1 */
    float sum_l = 0.0f;
    for (int i = 0; i < 4; i++)
        sum_l += comb_process(&r->comb[i], filtered);
    sum_l *= 0.25f;
    sum_l = allpass_process(&r->ap[0], sum_l);
    sum_l = allpass_process(&r->ap[1], sum_l);

    /* Right channel: combs 4-7 + allpass 2-3 (different primes = decorrelated) */
    float sum_r = 0.0f;
    for (int i = 4; i < 8; i++)
        sum_r += comb_process(&r->comb[i], filtered);
    sum_r *= 0.25f;
    sum_r = allpass_process(&r->ap[2], sum_r);
    sum_r = allpass_process(&r->ap[3], sum_r);

    *out_l = sum_l;
    *out_r = sum_r;
}

/* ── Delay (stereo ping-pong with tone filter) ── */
#define DLY_MAX_SAMPLES 44100  /* 1 second max per channel */

typedef struct {
    float buf_l[DLY_MAX_SAMPLES];
    float buf_r[DLY_MAX_SAMPLES];
    int   write_idx;
    float mix;          /* 0..1 dry/wet */
    float rate;         /* 0..1 -> 10ms..1000ms */
    float feedback;     /* 0..1 */
    float tone;         /* 0..1 (0=dark, 0.5=clean, 1=bright) */
    float lp_state_l;   /* feedback LPF state L */
    float hp_state_l;   /* feedback HPF state L */
    float lp_state_r;   /* feedback LPF state R */
    float hp_state_r;   /* feedback HPF state R */
    float smooth_time;  /* smoothed delay time in samples */
} wd_delay_t;

static void delay_init(wd_delay_t *d) {
    memset(d, 0, sizeof(*d));
    d->mix = 0.0f;
    d->rate = 0.3f;     /* ~300ms default */
    d->feedback = 0.3f;
    d->tone = 0.5f;     /* clean default */
}

static float delay_read(float *buf, int write_idx, float smooth_time) {
    float read_pos = (float)write_idx - smooth_time;
    if (read_pos < 0) read_pos += DLY_MAX_SAMPLES;
    int idx0 = (int)read_pos;
    float frac = read_pos - idx0;
    int idx1 = (idx0 + 1) % DLY_MAX_SAMPLES;
    idx0 = idx0 % DLY_MAX_SAMPLES;
    return buf[idx0] * (1.0f - frac) + buf[idx1] * frac;
}

static float delay_tone_filter(float signal, float tone, float *lp_state, float *hp_state) {
    if (tone < 0.48f) {
        float coeff = 0.05f + tone * 1.5f;
        *lp_state += coeff * (signal - *lp_state);
        return *lp_state;
    } else if (tone > 0.52f) {
        float coeff = 0.1f + (1.0f - tone) * 1.5f;
        *hp_state += coeff * (signal - *hp_state);
        return signal - *hp_state;
    }
    return signal;
}

static void delay_process_stereo(wd_delay_t *d, float in, float *out_l, float *out_r) {
    /* Rate: exponential 10ms..1000ms */
    float delay_ms = 10.0f * powf(100.0f, d->rate);
    float delay_samples = delay_ms * (SAMPLE_RATE / 1000.0f);
    if (delay_samples >= DLY_MAX_SAMPLES) delay_samples = DLY_MAX_SAMPLES - 1;

    /* Smooth delay time to avoid clicks */
    d->smooth_time += 0.001f * (delay_samples - d->smooth_time);

    /* Read from both delay lines */
    float del_l = delay_read(d->buf_l, d->write_idx, d->smooth_time);
    float del_r = delay_read(d->buf_r, d->write_idx, d->smooth_time);

    /* Tone filter on each feedback path */
    float fb_l = delay_tone_filter(del_l, d->tone, &d->lp_state_l, &d->hp_state_l);
    float fb_r = delay_tone_filter(del_r, d->tone, &d->lp_state_r, &d->hp_state_r);

    /* Ping-pong: input goes to L, L feedback crosses to R, R crosses to L */
    float fb = clampf(d->feedback, 0.0f, 0.95f);
    d->buf_l[d->write_idx] = in      + fb_r * fb;  /* input + R→L cross */
    d->buf_r[d->write_idx] = fb_l * fb;             /* L→R cross only */
    d->write_idx = (d->write_idx + 1) % DLY_MAX_SAMPLES;

    *out_l = del_l;
    *out_r = del_r;
}

/* ============================================================================
 * Instance
 * ============================================================================ */

/* ── 30 Kit presets: each defines 8 voice preset indices ── */
#define NUM_KITS 64
static const int KIT_PRESETS[NUM_KITS][8] = {
    /*  Kick  Snare Tom   Clap  Perc  HH-C  HH-O  Cymbal       */
    /* ── Classic machines ── */
    /* 0  Default */     { 0, 5, 10, 25, 30, 15, 16, 20 },
    /* 1  808 */         { 0, 5, 10, 25, 31, 15, 16, 20 },
    /* 2  909 */         { 1, 6, 11, 26, 30, 15, 16, 20 },
    /* 3  LinnDrum */    { 1, 5, 11, 25, 34, 15, 16, 21 },
    /* 4  CR-78 */       { 4, 9, 14, 29, 31, 17, 16, 23 },
    /* ── Genre kits ── */
    /* 5  Techno */      { 1, 6, 13, 25, 30, 15, 16, 20 },
    /* 6  House */       { 0, 5, 14, 25, 33, 15, 16, 20 },
    /* 7  Electro */     { 2, 9, 12, 26, 32, 18, 16, 21 },
    /* 8  Hip Hop */     { 0, 7, 14, 27, 30, 15, 16, 23 },
    /* 9  Trap */        { 0, 6, 11, 25, 30, 15, 16, 24 },
    /* 10 DnB */         { 1, 8, 12, 29, 34, 15, 16, 20 },
    /* 11 Dub */         { 4, 7, 10, 27, 31, 17, 16, 23 },
    /* 12 Garage */      { 0, 6, 11, 25, 34, 15, 16, 20 },
    /* 13 Jungle */      { 1, 8, 12, 26, 30, 15, 16, 20 },
    /* 14 Breakbeat */   { 1, 7, 11, 26, 30, 15, 16, 21 },
    /* 15 Reggaeton */   { 0, 6, 14, 25, 31, 15, 16, 20 },
    /* 16 Afrobeat */    { 0, 5, 14, 25, 31, 15, 17, 24 },
    /* 17 Disco */       { 0, 5, 10, 25, 33, 15, 16, 21 },
    /* 18 Funk */        { 1, 5, 10, 25, 32, 15, 16, 20 },
    /* 19 New Wave */    { 2, 6, 12, 26, 35, 18, 16, 21 },
    /* 20 Synthwave */   { 0, 6, 13, 25, 32, 15, 16, 24 },
    /* 21 EBM */         { 2, 8, 13, 28, 32, 18, 19, 21 },
    /* 22 Footwork */    { 1, 8, 11, 26, 30, 15, 15, 38 },
    /* 23 Gqom */        { 0, 9, 10, 28, 30, 19, 16, 20 },
    /* 24 Jersey Club */ { 1, 6, 12, 25, 30, 15, 16, 38 },
    /* ── Character kits ── */
    /* 25 Minimal */     { 1, 5, 10, 29, 32, 15, 16, 24 },
    /* 26 Industrial */  { 2, 8, 13, 28, 31, 18, 19, 21 },
    /* 27 Lo-Fi */       { 3, 7, 10, 27, 33, 19, 17, 23 },
    /* 28 Ambient */     { 4, 9, 14, 27, 33, 17, 16, 24 },
    /* 29 Noise */       { 3, 9, 13, 28, 39, 19, 19, 36 },
    /* 30 Glitch */      { 2, 8, 13, 28, 35, 18, 19, 36 },
    /* 31 Metal */       { 2, 8, 13, 28, 31, 18, 19, 22 },
    /* 32 Organic */     { 4, 7, 14, 27, 33, 17, 16, 23 },
    /* 33 Bright */      { 1, 6, 12, 26, 32, 15, 16, 21 },
    /* 34 Dark */        { 0, 7, 10, 27, 31, 19, 17, 23 },
    /* 35 Fast */        { 1, 6, 12, 26, 30, 15, 15, 38 },
    /* 36 Weird */       { 2, 8, 13, 28, 35, 18, 19, 36 },
    /* 37 Dusty */       { 3, 7, 14, 27, 33, 19, 17, 23 },
    /* 38 Crispy */      { 1, 8, 12, 26, 32, 18, 15, 22 },
    /* 39 Heavy */       { 2, 7, 10, 27, 31, 19, 19, 23 },
    /* 40 Tight */       { 1, 6, 12, 26, 30, 15, 15, 21 },
    /* 41 Loose */       { 4, 7, 14, 27, 34, 17, 16, 24 },
    /* 42 Saturated */   { 2, 8, 13, 28, 31, 18, 19, 22 },
    /* 43 Clean */       { 0, 5, 10, 29, 33, 15, 16, 20 },
    /* 44 Punchy */      { 1, 6, 11, 26, 30, 15, 16, 20 },
    /* 45 Muted */       { 3, 9, 10, 27, 33, 17, 17, 23 },
    /* 46 Fizzy */       { 2, 9, 12, 28, 34, 18, 16, 24 },
    /* 47 Thump */       { 0, 7, 10, 25, 31, 19, 17, 23 },
    /* 48 Micro */       { 1, 6, 12, 29, 32, 15, 15, 38 },
    /* ── Specialized kits ── */
    /* 49 Perc Only */   { 30, 31, 32, 33, 34, 14, 11, 12 },
    /* 50 FX Only */     { 35, 36, 37, 38, 39, 35, 38, 37 },
    /* 51 Clap Lab */    { 25, 26, 27, 28, 29, 25, 26, 27 },
    /* 52 Tom Rack */    { 10, 11, 12, 13, 14, 10, 11, 12 },
    /* 53 HH Stack */    { 15, 16, 17, 18, 19, 15, 16, 17 },
    /* 54 Cymbal Wash */ { 20, 21, 22, 23, 24, 20, 21, 22 },
    /* 55 Kick Army */   { 0, 1, 2, 3, 4, 0, 1, 2 },
    /* 56 Snare Rack */  { 5, 6, 7, 8, 9, 5, 6, 7 },
    /* ── Hybrid / crossover kits ── */
    /* 57 Kick+Perc */   { 0, 1, 30, 31, 32, 33, 34, 14 },
    /* 58 Snare+FX */    { 5, 6, 7, 8, 35, 36, 38, 39 },
    /* 59 Tom+Cymbal */  { 10, 11, 12, 20, 21, 22, 23, 24 },
    /* 60 808+FX */      { 0, 5, 10, 25, 35, 15, 36, 37 },
    /* 61 909+Perc */    { 1, 6, 11, 26, 30, 31, 32, 34 },
    /* 62 Drone Kit */   { 4, 37, 37, 39, 37, 17, 16, 24 },
    /* 63 Chaos */       { 35, 36, 37, 38, 39, 28, 36, 35 },
};
static const char *KIT_NAMES[NUM_KITS] = {
    "Default", "808", "909", "LinnDrum", "CR-78",
    "Techno", "House", "Electro", "Hip Hop", "Trap",
    "DnB", "Dub", "Garage", "Jungle", "Breakbeat",
    "Reggaetn", "Afrobeat", "Disco", "Funk", "New Wave",
    "Synthwav", "EBM", "Footwork", "Gqom", "JrsyClub",
    "Minimal", "Industrl", "Lo-Fi", "Ambient", "Noise",
    "Glitch", "Metal", "Organic", "Bright", "Dark",
    "Fast", "Weird", "Dusty", "Crispy", "Heavy",
    "Tight", "Loose", "Saturatn", "Clean", "Punchy",
    "Muted", "Fizzy", "Thump", "Micro",
    "PercOnly", "FX Only", "Clap Lab", "Tom Rack", "HH Stack",
    "CymbWash", "KickArmy", "SnrRack",
    "Kick+Prc", "Snare+FX", "Tom+Cymb", "808+FX", "909+Perc",
    "DroneKit", "Chaos"
};

typedef struct {
    wd_voice_t  voice[NUM_VOICES];
    wd_master_t master;
    float       voice_vol[NUM_VOICES];   /* mixer page volumes */
    float       voice_vol_smooth[NUM_VOICES];
    float       voice_pan_smooth[NUM_VOICES]; /* smoothed pan */
    int         current_page;            /* 0=patch, 1=general, 2=voice */
    int         current_voice;           /* 0..7, auto-selected from pad MIDI */
    int         midi_voice_cursor;       /* round-robin for MIDI trigger */
    int         current_kit;             /* 0..29 kit preset index */
    float       same_freq;               /* 0=off, >0 = master freq override (20..20000) */
    int         current_pitch_scale;     /* last applied pitch scale index (-1=none) */
    uint32_t    rng_state;               /* RNG for randomize */
    wd_reverb_t reverb;
    wd_delay_t  delay;
    /* Kit persistence */
    int         custom_presets[NUM_KITS][NUM_VOICES]; /* user-saved voice presets per kit */
    float       custom_vols[NUM_KITS][NUM_VOICES];    /* user-saved volumes per kit */
    float       custom_pans[NUM_KITS][NUM_VOICES];    /* user-saved pans per kit */
    int         save_kit_state;                       /* 0=Play, 1=Save (self-resetting) */
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
    v->reverb_send = inst_random(inst) * 0.5f;
    v->delay_send = inst_random(inst) * 0.3f;
}

static void randomize_patch(wd_instance_t *inst) {
    for (int i = 0; i < NUM_VOICES; i++) {
        randomize_voice(inst, i);
        inst->voice_vol[i] = 0.5f + inst_random(inst) * 0.4f;
    }
}

/* ── Random Pitch: 96 musically-tuned pitch scales ──
 * 8 categories × 12 roots (all chromatic notes in every category).
 * Each scale is generated from an interval pattern transposed by root.
 * On trigger, each voice gets its note placed in the octave
 * that fits its preset category's frequency range.
 */
#define NUM_PITCH_SCALES 96
#define NUM_SCALE_PATTERNS 15

/* Interval patterns: 8 semitone offsets from root for 8 voices.
 * 7-note scales repeat the root; 5-6 note scales repeat key tones. */
static const int SCALE_PATTERNS[NUM_SCALE_PATTERNS][8] = {
    {0,2,4,5,7,9,11,0},     /*  0: Major (Ionian)      */
    {0,2,3,5,7,8,10,0},     /*  1: Natural minor       */
    {0,2,4,7,9,0,4,7},      /*  2: Pentatonic major    */
    {0,3,5,7,10,0,5,7},     /*  3: Pentatonic minor    */
    {0,3,5,6,7,10,3,7},     /*  4: Blues                */
    {0,2,3,5,7,9,10,0},     /*  5: Dorian              */
    {0,1,3,5,7,8,10,0},     /*  6: Phrygian            */
    {0,2,4,6,7,9,11,0},     /*  7: Lydian              */
    {0,2,4,5,7,9,10,0},     /*  8: Mixolydian          */
    {0,2,3,5,7,8,11,0},     /*  9: Harmonic minor      */
    {0,2,3,5,7,9,11,0},     /* 10: Melodic minor       */
    {0,2,4,6,8,10,0,6},     /* 11: Whole tone          */
    {0,2,3,5,6,8,9,11},     /* 12: Diminished (h-w)    */
    {0,1,4,5,7,8,11,0},     /* 13: Hungarian minor     */
    {0,1,5,7,10,0,5,7},     /* 14: Japanese In         */
};

static const char *PATTERN_NAMES[NUM_SCALE_PATTERNS] = {
    "Maj","min","M.Pn","m.Pn","Blue",
    "Dori","Phry","Lydi","Mixo",
    "H.mn","Ml.m",
    "Whol","Dimn","Hung","Japn",
};

static const char *ROOT_NAMES[12] = {
    "C","Db","D","Eb","E","F","F#","G","Ab","A","Bb","B"
};

/* 8 categories × 12 roots = 96 scales.
 * Categories with sub-types cycle through them across roots:
 *   Modes:   Dorian→Phrygian→Lydian→Mixolydian (3 roots each)
 *   Har/Mel: Harmonic→Melodic (6 roots each)
 *   Exotic:  WholeTone→Diminished→Hungarian→Japanese (3 roots each) */
static void scale_info(int scale_idx, int *out_pattern, int *out_root) {
    int cat = scale_idx / 12;
    int ri = scale_idx % 12;
    *out_root = ri;
    switch (cat) {
        case 0: *out_pattern = 0;  break;            /* Major       */
        case 1: *out_pattern = 1;  break;            /* Minor       */
        case 2: *out_pattern = 2;  break;            /* Pent Major  */
        case 3: *out_pattern = 3;  break;            /* Pent Minor  */
        case 4: *out_pattern = 4;  break;            /* Blues        */
        case 5: *out_pattern = 5 + (ri % 4);  break; /* Modes       */
        case 6: *out_pattern = 9 + (ri % 2);  break; /* Harm/Mel    */
        case 7: *out_pattern = 11 + (ri % 4); break; /* Exotic      */
        default: *out_pattern = 0; break;
    }
}

/* MIDI range per preset category — based on real hardware drum synth tuning:
 *   808/909 kick: 30-80Hz, RYTM/Nord Drum: 20-200Hz, DRM1: 35-200Hz
 *   808 snare: 150-350Hz, 909: 150-400Hz, RYTM: 80-400Hz, Nord: 100-500Hz
 *   808 toms: 80-400Hz across low/mid/high, RYTM: 40-400Hz
 *   808 hats: metallic osc 200-800Hz, RYTM: 200-800Hz
 *   Cymbals: tone component 150-800Hz
 *   808 clap: noise-based 200-1200Hz tonal body
 *   Perc (cowbell 560-800, rimshot 200-800, clave 2000-3000): wide 150-1600Hz
 *   FX: full synth range 40-1600Hz
 */
static void category_midi_range(int preset, int *out_min, int *out_max) {
    if (preset <= 4)       { *out_min = 24; *out_max = 47; }  /* Kicks:  C1-B2  (32-123Hz)  */
    else if (preset <= 9)  { *out_min = 40; *out_max = 67; }  /* Snares: E2-G4  (82-392Hz)  */
    else if (preset <= 14) { *out_min = 31; *out_max = 65; }  /* Toms:   G1-F4  (49-349Hz)  */
    else if (preset <= 19) { *out_min = 55; *out_max = 79; }  /* Hats:   G3-G5  (196-784Hz) */
    else if (preset <= 24) { *out_min = 50; *out_max = 79; }  /* Cymb:   D3-G5  (147-784Hz) */
    else if (preset <= 29) { *out_min = 55; *out_max = 84; }  /* Claps:  G3-C6  (196-1047Hz)*/
    else if (preset <= 34) { *out_min = 50; *out_max = 88; }  /* Perc:   D3-E6  (147-1319Hz)*/
    else if (preset <= 39) { *out_min = 28; *out_max = 88; }  /* FX:     E1-E6  (41-1319Hz) */
    else                   { *out_min = 36; *out_max = 79; }  /* Custom: C2-G5  (65-784Hz)  */
}

static float midi_to_freq(int note) {
    return 440.0f * powf(2.0f, (float)(note - 69) / 12.0f);
}

/* Find all octaves where this semitone falls within [min_midi, max_midi],
 * then pick one randomly for variety across repeated presses */
static float fit_semitone_to_range(int semitone, int min_midi, int max_midi, wd_instance_t *inst) {
    int candidates[8];
    int count = 0;
    for (int oct = 0; oct < 9; oct++) {
        int midi = oct * 12 + semitone;
        if (midi >= min_midi && midi <= max_midi)
            candidates[count++] = midi;
    }
    if (count == 0)
        return midi_to_freq((min_midi + max_midi) / 2);
    int idx = (int)(inst_random(inst) * (float)count) % count;
    return midi_to_freq(candidates[idx]);
}

/* Noise filter cutoff ratio per category — derived from preset freq:cutoff averages.
 * Kicks: 38→250(6.6x), 65→600(9.2x), 50→500(10x) ≈ 8x
 * Snares: 190→3500(18x), 220→4000(18x), 150→2000(13x) ≈ 16x
 * Toms: 75→700(9x), 130→900(7x), 200→1200(6x) ≈ 7x
 * HH: 450→9000(20x), 420→6500(15x), 380→7000(18x) ≈ 16x
 * Cymbals: 280→4500(16x), 600→5500(9x), 250→3000(12x) ≈ 12x
 * Claps: 800→1800(2.3x), 700→1500(2.1x), 1200→3000(2.5x) ≈ 2.5x
 * Perc: 500→4000(8x), 560→2200(4x), 2500→3000(1.2x) ≈ 4x
 * FX: 1200→8000(6.7x), 320→3000(9.4x), 60→1500(25x) ≈ 6x */
static float category_noise_cutoff_ratio(int preset) {
    if (preset <= 4)  return 8.0f;   /* Kicks  */
    if (preset <= 9)  return 16.0f;  /* Snares */
    if (preset <= 14) return 7.0f;   /* Toms   */
    if (preset <= 19) return 16.0f;  /* HH     */
    if (preset <= 24) return 12.0f;  /* Cymbals*/
    if (preset <= 29) return 2.5f;   /* Claps  */
    if (preset <= 34) return 4.0f;   /* Perc   */
    if (preset <= 39) return 6.0f;   /* FX     */
    return 6.0f;                     /* Custom */
}

static void randomize_pitch(wd_instance_t *inst) {
    int si = (int)(inst_random(inst) * NUM_PITCH_SCALES) % NUM_PITCH_SCALES;
    inst->current_pitch_scale = si;
    int pat, root;
    scale_info(si, &pat, &root);
    for (int i = 0; i < NUM_VOICES; i++) {
        int semitone = (SCALE_PATTERNS[pat][i] + root) % 12;
        int min_m, max_m;
        category_midi_range(inst->voice[i].preset, &min_m, &max_m);
        float f = fit_semitone_to_range(semitone, min_m, max_m, inst);
        inst->voice[i].freq = f;
        float ratio = category_noise_cutoff_ratio(inst->voice[i].preset);
        inst->voice[i].filter_cutoff = clampf(f * ratio, 20.0f, 18000.0f);
    }
}

static void apply_kit(wd_instance_t *inst, int kit) {
    if (kit < 0 || kit >= NUM_KITS) return;
    inst->current_kit = kit;
    for (int i = 0; i < NUM_VOICES; i++) {
        voice_apply_preset(&inst->voice[i], inst->custom_presets[kit][i]);
        inst->voice_vol[i] = inst->custom_vols[kit][i];
        inst->voice[i].pan = inst->custom_pans[kit][i];
    }
}

/* ── MIDI note to voice mapping ──
 * Pads on Move send notes. We map pad positions to voices:
 *   Voice 0-7 triggered by notes: any note triggers round-robin,
 *   OR fixed mapping: notes 36-43 (C2-G#2) = voices 0-7
 */
#define MIDI_NOTE_BASE 36

/* ── Page/Knob mapping tables ── */

/* General page: crush, filter, 3-band EQ (gain+freq paired) */
static const char *GENERAL_KNOB_NAMES[8] = {
    "Crush", "Filter", "Lo Gain", "Lo Freq",
    "Mid Gain", "Mid Freq", "Hi Gain", "Hi Freq"
};

/* Dynamic voice page: vol, pan, freq, decay, wave, mix, cutoff, preset */
static const char *VOICE_KNOB_NAMES[8] = {
    "Volume", "Pan", "Freq", "Decay",
    "Wave", "Mix", "Cutoff", "Preset"
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
 * Kit persistence — binary save/load to /data partition
 * ============================================================================ */

#define WD_KITS_FILE   "/data/UserData/schwung/weird_dreams_kits.dat"
#define WD_KITS_MAGIC  0x57444B54u  /* 'WDKT' */
#define WD_KITS_VER    1u

typedef struct {
    uint32_t magic, version, count, reserved;
    int   presets[NUM_KITS][NUM_VOICES];
    float vols[NUM_KITS][NUM_VOICES];
    float pans[NUM_KITS][NUM_VOICES];
} wd_kits_file_t;

static void wd_save_kits(wd_instance_t *inst) {
    FILE *f = fopen(WD_KITS_FILE, "wb");
    if (!f) return;
    wd_kits_file_t file;
    file.magic = WD_KITS_MAGIC; file.version = WD_KITS_VER;
    file.count = NUM_KITS; file.reserved = 0;
    memcpy(file.presets, inst->custom_presets, sizeof(file.presets));
    memcpy(file.vols,    inst->custom_vols,    sizeof(file.vols));
    memcpy(file.pans,    inst->custom_pans,    sizeof(file.pans));
    fwrite(&file, sizeof(file), 1, f);
    fclose(f);
}

static void wd_load_kits(wd_instance_t *inst) {
    FILE *f = fopen(WD_KITS_FILE, "rb");
    if (!f) return;
    wd_kits_file_t file;
    int ok = (fread(&file, sizeof(file), 1, f) == 1);
    fclose(f);
    if (!ok || file.magic != WD_KITS_MAGIC || file.version != WD_KITS_VER || file.count != NUM_KITS) return;
    memcpy(inst->custom_presets, file.presets, sizeof(file.presets));
    memcpy(inst->custom_vols,    file.vols,    sizeof(file.vols));
    memcpy(inst->custom_pans,    file.pans,    sizeof(file.pans));
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
    inst->current_voice = 0;
    inst->midi_voice_cursor = 0;
    inst->current_kit = 0;
    inst->same_freq = 0.0f;
    /* Initialize custom kits from built-in defaults */
    for (int k = 0; k < NUM_KITS; k++) {
        for (int v = 0; v < NUM_VOICES; v++) {
            inst->custom_presets[k][v] = KIT_PRESETS[k][v];
            /* Default volumes/pans from voice preset defaults */
            wd_voice_t tmp; voice_init(&tmp, v);
            voice_apply_preset(&tmp, KIT_PRESETS[k][v]);
            inst->custom_vols[k][v] = tmp.level;
            inst->custom_pans[k][v] = tmp.pan;
        }
    }
    wd_load_kits(inst);
    inst->save_kit_state = 0;
    inst->current_pitch_scale = -1;
    inst->rng_state = 987654321u;

    reverb_init(&inst->reverb);
    reverb_set_type(&inst->reverb, 0);
    reverb_update_params(&inst->reverb);
    delay_init(&inst->delay);

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

    if (status == 0xB0) {
        /* MIDI CC — per-voice freq and decay from upstream sequencer */
        uint8_t cc  = msg[1];
        uint8_t val = msg[2];
        float norm  = (float)val / 127.0f;  /* 0..1 */

        if (cc >= 70 && cc <= 77) {
            /* CC 70-77 → voice 0-7 frequency (20..20000 Hz exponential) */
            int vi = cc - 70;
            inst->voice[vi].freq = knob_to_freq(norm);
        } else if (cc >= 80 && cc <= 87) {
            /* CC 80-87 → voice 0-7 decay (0.0001..4.0 sec exponential) */
            int vi = cc - 80;
            inst->voice[vi].decay = knob_to_decay(norm);
        }
    } else if (status == 0x90 && vel > 0) {
        /* Note On — map to voice */
        int voice_idx;
        if (note >= MIDI_NOTE_BASE && note < MIDI_NOTE_BASE + NUM_VOICES) {
            /* Lower pads: C2=voice0, C#2=voice1, ... G#2=voice7 */
            voice_idx = note - MIDI_NOTE_BASE;
        } else if (note >= MIDI_NOTE_BASE + NUM_VOICES && note < MIDI_NOTE_BASE + NUM_VOICES * 2) {
            /* Upper pads mirror lower: notes 44-51 → voices 0-7 */
            voice_idx = note - MIDI_NOTE_BASE - NUM_VOICES;
        } else {
            /* Other notes: round-robin */
            voice_idx = inst->midi_voice_cursor;
            inst->midi_voice_cursor = (inst->midi_voice_cursor + 1) % NUM_VOICES;
        }
        voice_trigger(&inst->voice[voice_idx], (float)vel / 127.0f);

        /* Pad selects current voice for dynamic Voice page */
        inst->current_voice = voice_idx;
        inst->current_page = 2; /* Voice page */
    }
}

/* ── set_param ── */
static void set_param(void *instance, const char *key, const char *val) {
    wd_instance_t *inst = (wd_instance_t *)instance;
    if (!inst || !key || !val) return;

    /* Page switching: 0=patch, 1=general, 2=voice */
    if (strcmp(key, "_level") == 0) {
        if (strcmp(val, "Patch") == 0) inst->current_page = 0;
        else if (strcmp(val, "General") == 0) inst->current_page = 1;
        else if (strcmp(val, "Voice") == 0) inst->current_page = 2;
        else if (strcmp(val, "FX") == 0) inst->current_page = 3;
        return;
    }

    /* Knob adjust */
    if (strncmp(key, "knob_", 5) == 0 && strstr(key, "_adjust")) {
        int knob = atoi(key + 5) - 1;
        if (knob < 0 || knob > 7) return;
        int delta = atoi(val);
        int page = inst->current_page;

        if (page == 0) {
            /* Patch page: Kit, Rnd Kit, Rnd Voice, Rnd Pitch, SameFreq, Init Freq, Rnd Pan, All Mono */
            switch (knob) {
                case 0: { /* Kit (jog) */
                    inst->current_kit = (inst->current_kit + (delta > 0 ? 1 : -1) + NUM_KITS) % NUM_KITS;
                    apply_kit(inst, inst->current_kit);
                } break;
                case 1: /* Rnd Kit — randomize entire kit */
                    if (delta != 0) randomize_patch(inst);
                    break;
                case 2: /* Rnd Voice — randomize the current voice */
                    if (delta != 0) randomize_voice(inst, inst->current_voice);
                    break;
                case 3: /* Rnd Pitch — apply a random musical scale to all voices */
                    if (delta != 0) randomize_pitch(inst);
                    break;
                case 4: { /* SameFreq — master frequency + filter cutoff for all voices */
                    float k = inst->same_freq > 0.0f ? freq_to_knob(inst->same_freq) : 0.5f;
                    k = clampf(k + delta * 0.005f, 0.0f, 1.0f);
                    inst->same_freq = knob_to_freq(k);
                    for (int i = 0; i < NUM_VOICES; i++) {
                        inst->voice[i].freq = inst->same_freq;
                        inst->voice[i].filter_cutoff = clampf(inst->same_freq, 20.0f, 18000.0f);
                    }
                } break;
                case 5: /* Init Freq — restore kit frequencies + cutoffs */
                    if (delta != 0) {
                        inst->same_freq = 0.0f;
                        int kit = inst->current_kit;
                        for (int i = 0; i < NUM_VOICES; i++) {
                            wd_voice_t tmp;
                            memset(&tmp, 0, sizeof(tmp));
                            voice_apply_preset(&tmp, KIT_PRESETS[kit][i]);
                            inst->voice[i].freq = tmp.freq;
                            inst->voice[i].filter_cutoff = tmp.filter_cutoff;
                        }
                    } break;
                case 6: /* Rnd Pan — randomize panning (kicks stay center) */
                    if (delta != 0) {
                        for (int i = 0; i < NUM_VOICES; i++) {
                            if (inst->voice[i].preset <= 4)
                                inst->voice[i].pan = 0.0f;
                            else
                                inst->voice[i].pan = (inst_random(inst) * 2.0f - 1.0f) * 0.8f;
                        }
                    } break;
                case 7: /* All Mono — reset all panning to center */
                    if (delta != 0) {
                        for (int i = 0; i < NUM_VOICES; i++)
                            inst->voice[i].pan = 0.0f;
                    } break;
            }
        } else if (page == 1) {
            /* General page: comp, filter, 3-band EQ (gain+freq paired) */
            switch (knob) {
                case 0: inst->master.comp_amount = clampf(inst->master.comp_amount + delta * 0.01f, 0.0f, 1.0f); break;
                case 1: inst->master.dj_filter = clampf(inst->master.dj_filter + delta * 0.005f, 0.0f, 1.0f); break;
                case 2: inst->master.eq_low_gain = clampf(inst->master.eq_low_gain + delta * 0.24f, -12.0f, 12.0f); break;
                case 3: inst->master.eq_low_freq = clampf(inst->master.eq_low_freq + delta * 4.8f, 20.0f, 500.0f); break;
                case 4: inst->master.eq_mid_gain = clampf(inst->master.eq_mid_gain + delta * 0.24f, -12.0f, 12.0f); break;
                case 5: inst->master.eq_mid_freq = clampf(inst->master.eq_mid_freq + delta * 80.0f, 200.0f, 8000.0f); break;
                case 6: inst->master.eq_high_gain = clampf(inst->master.eq_high_gain + delta * 0.24f, -12.0f, 12.0f); break;
                case 7: inst->master.eq_high_freq = clampf(inst->master.eq_high_freq + delta * 160.0f, 2000.0f, 18000.0f); break;
            }
        } else if (page == 3) {
            /* FX page */
            switch (knob) {
                case 0: inst->reverb.mix = clampf(inst->reverb.mix + delta * 0.01f, 0.0f, 1.0f); break;
                case 1: {
                    inst->reverb.type = (inst->reverb.type + (delta > 0 ? 1 : -1) + REV_NUM_TYPES) % REV_NUM_TYPES;
                    reverb_set_type(&inst->reverb, inst->reverb.type);
                    reverb_update_params(&inst->reverb);
                } break;
                case 2:
                    inst->reverb.size = clampf(inst->reverb.size + delta * 0.01f, 0.0f, 1.0f);
                    reverb_set_type(&inst->reverb, inst->reverb.type);
                    reverb_update_params(&inst->reverb);
                    break;
                case 3:
                    inst->reverb.decay = clampf(inst->reverb.decay + delta * 0.01f, 0.0f, 1.0f);
                    reverb_update_params(&inst->reverb);
                    break;
                case 4: inst->delay.mix = clampf(inst->delay.mix + delta * 0.01f, 0.0f, 1.0f); break;
                case 5: inst->delay.rate = clampf(inst->delay.rate + delta * 0.01f, 0.0f, 1.0f); break;
                case 6: inst->delay.feedback = clampf(inst->delay.feedback + delta * 0.01f, 0.0f, 0.95f); break;
                case 7: inst->delay.tone = clampf(inst->delay.tone + delta * 0.01f, 0.0f, 1.0f); break;
            }
        } else {
            /* Voice page (dynamic): Volume, Pan, Freq, Decay, Wave, Mix, Cutoff, Preset */
            int vi = inst->current_voice;
            wd_voice_t *v = &inst->voice[vi];
            switch (knob) {
                case 0: /* Volume */
                    inst->voice_vol[vi] = clampf(inst->voice_vol[vi] + delta * 0.01f, 0.0f, 1.0f);
                    break;
                case 1: /* Pan */
                    v->pan = clampf(v->pan + delta * 0.02f, -1.0f, 1.0f);
                    break;
                case 2: { /* Freq (exponential) */
                    float k = freq_to_knob(v->freq);
                    k = clampf(k + delta * 0.005f, 0.0f, 1.0f);
                    v->freq = knob_to_freq(k);
                } break;
                case 3: { /* Decay (exponential) */
                    float k = decay_to_knob(v->decay);
                    k = clampf(k + delta * 0.01f, 0.0f, 1.0f);
                    v->decay = knob_to_decay(k);
                } break;
                case 4: /* Wave (continuous morph) */
                    v->wave = clampf(v->wave + delta * 0.01f, 0.0f, 1.0f);
                    break;
                case 5: /* Mix */
                    v->mix = clampf(v->mix + delta * 0.01f, 0.0f, 1.0f);
                    break;
                case 6: { /* Cutoff (exponential) */
                    float k = cutoff_to_knob(v->filter_cutoff);
                    k = clampf(k + delta * 0.005f, 0.0f, 1.0f);
                    v->filter_cutoff = knob_to_cutoff(k);
                } break;
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
    if (strcmp(key, "rnd_voice") == 0) { if (f != 0) randomize_voice(inst, inst->current_voice); return; }
    if (strcmp(key, "rnd_kit") == 0) { if (f != 0) randomize_patch(inst); return; }
    if (strcmp(key, "rnd_pitch") == 0) { if (f != 0) randomize_pitch(inst); return; }
    if (strcmp(key, "init_freq") == 0) {
        if (f != 0) {
            inst->same_freq = 0.0f;
            int kit = inst->current_kit;
            for (int i=0;i<NUM_VOICES;i++) { wd_voice_t tmp; memset(&tmp,0,sizeof(tmp)); voice_apply_preset(&tmp,KIT_PRESETS[kit][i]); inst->voice[i].freq=tmp.freq; inst->voice[i].filter_cutoff=tmp.filter_cutoff; }
        }
        return;
    }
    if (strcmp(key, "all_mono") == 0) {
        if (f != 0) { for (int i=0;i<NUM_VOICES;i++) inst->voice[i].pan=0.0f; }
        return;
    }
    if (strcmp(key, "save_kit") == 0) {
        if (strcmp(val, "Save") == 0) {
            int k = inst->current_kit;
            for (int i = 0; i < NUM_VOICES; i++) {
                inst->custom_presets[k][i] = inst->voice[i].preset;
                inst->custom_vols[k][i]    = inst->voice_vol[i];
                inst->custom_pans[k][i]    = inst->voice[i].pan;
            }
            wd_save_kits(inst);
            inst->save_kit_state = 0;
        }
        return;
    }
    if (strcmp(key, "reset_eq") == 0) {
        if (f != 0) {
            inst->master.eq_low_gain = 0.0f; inst->master.eq_mid_gain = 0.0f; inst->master.eq_high_gain = 0.0f;
            inst->master.eq_low_freq = 200.0f; inst->master.eq_mid_freq = 1000.0f; inst->master.eq_high_freq = 8000.0f;
            inst->master.eq_low_q = 1.0f; inst->master.eq_mid_q = 1.0f; inst->master.eq_high_q = 1.0f;
        }
        return;
    }
    if (strcmp(key, "rnd_pan") == 0) {
        if (f != 0) { for (int i=0;i<NUM_VOICES;i++) { if (inst->voice[i].preset<=4) inst->voice[i].pan=0; else inst->voice[i].pan=(inst_random(inst)*2.0f-1.0f)*0.8f; } }
        return;
    }
    if (strcmp(key, "same_freq") == 0) { inst->same_freq = clampf(f, 20.0f, 20000.0f); for (int i=0;i<NUM_VOICES;i++) { inst->voice[i].freq = inst->same_freq; inst->voice[i].filter_cutoff = clampf(inst->same_freq, 20.0f, 18000.0f); } return; }
    if (strcmp(key, "master") == 0) { inst->master.master_level = clampf(f, 0.0f, 1.0f); return; }

    /* FX params */
    if (strcmp(key, "rev_mix") == 0) { inst->reverb.mix = clampf(f, 0.0f, 1.0f); return; }
    if (strcmp(key, "rev_type") == 0) {
        int t = -1;
        for (int i = 0; i < REV_NUM_TYPES; i++) { if (strcmp(val, REV_TYPE_NAMES[i]) == 0) { t = i; break; } }
        if (t < 0) t = (int)clampf(f, 0, REV_NUM_TYPES - 1);
        inst->reverb.type = t;
        reverb_set_type(&inst->reverb, t);
        reverb_update_params(&inst->reverb);
        return;
    }
    if (strcmp(key, "rev_size") == 0) { inst->reverb.size = clampf(f, 0.0f, 1.0f); reverb_set_type(&inst->reverb, inst->reverb.type); reverb_update_params(&inst->reverb); return; }
    if (strcmp(key, "rev_decay") == 0) { inst->reverb.decay = clampf(f, 0.0f, 1.0f); reverb_update_params(&inst->reverb); return; }
    if (strcmp(key, "dly_mix") == 0) { inst->delay.mix = clampf(f, 0.0f, 1.0f); return; }
    if (strcmp(key, "dly_rate") == 0) { inst->delay.rate = clampf(f, 0.0f, 1.0f); return; }
    if (strcmp(key, "dly_fdbk") == 0) { inst->delay.feedback = clampf(f, 0.0f, 0.95f); return; }
    if (strcmp(key, "dly_tone") == 0) { inst->delay.tone = clampf(f, 0.0f, 1.0f); return; }

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
        snprintf(k, sizeof(k), "v%d_rsend", i+1);
        if (strcmp(key, k) == 0) { v->reverb_send = clampf(f, 0.0f, 1.0f); return; }
        snprintf(k, sizeof(k), "v%d_dsend", i+1);
        if (strcmp(key, k) == 0) { v->delay_send = clampf(f, 0.0f, 1.0f); return; }
    }

    /* Virtual cv_* params — redirect to current voice */
    if (strncmp(key, "cv_", 3) == 0) {
        int vi = inst->current_voice;
        wd_voice_t *v = &inst->voice[vi];
        const char *suffix = key + 3;
        if (strcmp(suffix, "vol") == 0) { inst->voice_vol[vi] = clampf(f, 0.0f, 1.0f); return; }
        if (strcmp(suffix, "pan") == 0) { v->pan = clampf(f, -1.0f, 1.0f); return; }
        if (strcmp(suffix, "freq") == 0) { v->freq = clampf(f, 20.0f, 20000.0f); return; }
        if (strcmp(suffix, "decay") == 0) { v->decay = clampf(f, 0.0001f, 4.0f); return; }
        if (strcmp(suffix, "wave") == 0) {
            if (strcmp(val, "Sine") == 0) v->wave = 0.0f;
            else if (strcmp(val, "Triangle") == 0) v->wave = 0.33f;
            else if (strcmp(val, "Saw") == 0) v->wave = 0.66f;
            else if (strcmp(val, "Square") == 0) v->wave = 1.0f;
            else v->wave = clampf(f, 0.0f, 1.0f);
            return;
        }
        if (strcmp(suffix, "mix") == 0) { v->mix = clampf(f, 0.0f, 1.0f); return; }
        if (strcmp(suffix, "cutoff") == 0) { v->filter_cutoff = clampf(f, 20.0f, 18000.0f); return; }
        if (strcmp(suffix, "preset") == 0) {
            int p = -1;
            for (int j = 0; j < NUM_PRESETS; j++) {
                if (strcmp(val, PRESET_NAMES[j]) == 0) { p = j; break; }
            }
            if (p < 0) p = (int)f;
            if (p >= 0 && p < NUM_PRESETS - 1) voice_apply_preset(v, p);
            else v->preset = NUM_PRESETS - 1;
            return;
        }
        if (strcmp(suffix, "attack") == 0) { v->attack = clampf(f, 0.0001f, 1.0f); return; }
        if (strcmp(suffix, "penv") == 0) { v->pitch_env_amt = clampf(f, 0.0f, 1.0f); return; }
        if (strcmp(suffix, "prate") == 0) { v->pitch_env_rate = clampf(f, 0.001f, 2.0f); return; }
        if (strcmp(suffix, "lamt") == 0) { v->pitch_lfo_amt = clampf(f, 0.0f, 1.0f); return; }
        if (strcmp(suffix, "lrate") == 0) { v->pitch_lfo_rate = clampf(f, 0.1f, 80.0f); return; }
        if (strcmp(suffix, "ftype") == 0) {
            if (strcmp(val, "LP") == 0) v->filter_type = 0;
            else if (strcmp(val, "HP") == 0) v->filter_type = 1;
            else if (strcmp(val, "BP") == 0) v->filter_type = 2;
            else v->filter_type = (int)clampf(f, 0, 2);
            return;
        }
        if (strcmp(suffix, "fres") == 0) { v->filter_res = clampf(f, 1.0f, 5.0f); return; }
        if (strcmp(suffix, "nattack") == 0) { v->noise_attack = clampf(f, 0.0001f, 1.0f); return; }
        if (strcmp(suffix, "ndecay") == 0) { v->noise_decay = clampf(f, 0.0001f, 1.0f); return; }
        if (strcmp(suffix, "dist") == 0) { v->distortion = clampf(f, 0.0f, 50.0f); return; }
        if (strcmp(suffix, "level") == 0) { v->level = clampf(f, 0.0f, 1.0f); return; }
        if (strcmp(suffix, "rsend") == 0) { v->reverb_send = clampf(f, 0.0f, 1.0f); return; }
        if (strcmp(suffix, "dsend") == 0) { v->delay_send = clampf(f, 0.0f, 1.0f); return; }
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

        /* FX */
        NEXT_TOKEN(); inst->reverb.mix = atof(token);
        NEXT_TOKEN(); inst->reverb.type = atoi(token);
        NEXT_TOKEN(); inst->reverb.size = atof(token);
        NEXT_TOKEN(); inst->reverb.decay = atof(token);
        reverb_set_type(&inst->reverb, inst->reverb.type);
        reverb_update_params(&inst->reverb);
        NEXT_TOKEN(); inst->delay.mix = atof(token);
        NEXT_TOKEN(); inst->delay.rate = atof(token);
        NEXT_TOKEN(); inst->delay.feedback = atof(token);
        NEXT_TOKEN(); inst->delay.tone = atof(token);

        /* Per-voice: preset freq attack decay wave penv prate lamt lrate ftype cutoff fres nattack ndecay mix dist level pan rsend dsend */
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
            NEXT_TOKEN(); v->reverb_send = atof(token);
            NEXT_TOKEN(); v->delay_send = atof(token);
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
        return snprintf(buf, buf_len, "WDrms");

    /* Knob names */
    if (strncmp(key, "knob_", 5) == 0 && strstr(key, "_name")) {
        int knob = atoi(key + 5) - 1;
        if (knob < 0 || knob > 7) return -1;
        int page = inst->current_page;

        if (page == 0) {
            static const char *PATCH_N[8] = {"Kit","Rnd Kit","Rnd Voice","Rnd Pitch","SameFreq","Init Freq","Rnd Pan","All Mono"};
            return snprintf(buf, buf_len, "%s", PATCH_N[knob]);
        }
        if (page == 1) return snprintf(buf, buf_len, "%s", GENERAL_KNOB_NAMES[knob]);
        if (page == 3) {
            static const char *FX_N[8] = {"Rev Mix","Rev Type","Rev Size","Rev Decay","Dly Mix","Dly Rate","Dly Fdbk","Dly Tone"};
            return snprintf(buf, buf_len, "%s", FX_N[knob]);
        }
        /* Voice page (dynamic) — show voice number in name */
        return snprintf(buf, buf_len, "V%d %s", inst->current_voice + 1, VOICE_KNOB_NAMES[knob]);
    }

    /* Knob values */
    if (strncmp(key, "knob_", 5) == 0 && strstr(key, "_value")) {
        int knob = atoi(key + 5) - 1;
        if (knob < 0 || knob > 7) return -1;
        int page = inst->current_page;

        if (page == 0) {
            /* Patch page */
            switch (knob) {
                case 0: return snprintf(buf, buf_len, "%s", KIT_NAMES[inst->current_kit]);
                case 1: return snprintf(buf, buf_len, "Turn");
                case 2: return snprintf(buf, buf_len, "Turn");
                case 3: {
                    if (inst->current_pitch_scale >= 0) {
                        int pat, root;
                        scale_info(inst->current_pitch_scale, &pat, &root);
                        return snprintf(buf, buf_len, "%s %s", ROOT_NAMES[root], PATTERN_NAMES[pat]);
                    }
                    return snprintf(buf, buf_len, "Turn");
                }
                case 4: return snprintf(buf, buf_len, "%dHz", inst->same_freq > 0 ? (int)inst->same_freq : 0);
                case 5: return snprintf(buf, buf_len, "Turn");
                case 6: return snprintf(buf, buf_len, "Turn");
                case 7: return snprintf(buf, buf_len, "Turn");
            }
        }
        if (page == 1) {
            /* General page */
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
        if (page == 3) {
            switch (knob) {
                case 0: return snprintf(buf, buf_len, "%d%%", (int)(inst->reverb.mix * 100.0f));
                case 1: return snprintf(buf, buf_len, "%s", REV_TYPE_NAMES[inst->reverb.type]);
                case 2: return snprintf(buf, buf_len, "%d%%", (int)(inst->reverb.size * 100.0f));
                case 3: return snprintf(buf, buf_len, "%d%%", (int)(inst->reverb.decay * 100.0f));
                case 4: return snprintf(buf, buf_len, "%d%%", (int)(inst->delay.mix * 100.0f));
                case 5: {
                    float ms = 10.0f * powf(100.0f, inst->delay.rate);
                    return snprintf(buf, buf_len, "%dms", (int)ms);
                }
                case 6: return snprintf(buf, buf_len, "%d%%", (int)(inst->delay.feedback * 100.0f));
                case 7: {
                    float t = inst->delay.tone;
                    if (t < 0.48f) return snprintf(buf, buf_len, "Dark %d%%", (int)((0.5f - t) * 200.0f));
                    if (t > 0.52f) return snprintf(buf, buf_len, "Brite %d%%", (int)((t - 0.5f) * 200.0f));
                    return snprintf(buf, buf_len, "Clean");
                }
            }
        }
        /* Voice page (dynamic) — Volume, Pan, Freq, Decay, Wave, Mix, Cutoff, Preset */
        {
            int vi = inst->current_voice;
            wd_voice_t *v = &inst->voice[vi];
            switch (knob) {
                case 0: return snprintf(buf, buf_len, "%d%%", (int)(inst->voice_vol[vi] * 100.0f));
                case 1: {
                    float p = v->pan;
                    if (p < -0.01f) return snprintf(buf, buf_len, "L%d", (int)(-p * 100));
                    if (p > 0.01f) return snprintf(buf, buf_len, "R%d", (int)(p * 100));
                    return snprintf(buf, buf_len, "C");
                }
                case 2: return snprintf(buf, buf_len, "%dHz", (int)v->freq);
                case 3: return snprintf(buf, buf_len, "%dms", (int)(v->decay * 1000.0f));
                case 4: {
                    float w = v->wave * 3.0f;
                    if (w < 0.1f) return snprintf(buf, buf_len, "Sine");
                    if (w < 1.0f) return snprintf(buf, buf_len, "Si>Tri %d%%", (int)(w * 100));
                    if (w < 1.1f) return snprintf(buf, buf_len, "Tri");
                    if (w < 2.0f) return snprintf(buf, buf_len, "Tr>Saw %d%%", (int)((w-1)*100));
                    if (w < 2.1f) return snprintf(buf, buf_len, "Saw");
                    if (w < 3.0f) return snprintf(buf, buf_len, "Sw>Sqr %d%%", (int)((w-2)*100));
                    return snprintf(buf, buf_len, "Square");
                }
                case 5: return snprintf(buf, buf_len, "%d%%", (int)(v->mix * 100.0f));
                case 6: return snprintf(buf, buf_len, "%dHz", (int)v->filter_cutoff);
                case 7: return snprintf(buf, buf_len, "%s", PRESET_NAMES[v->preset]);
            }
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
            "{\"key\":\"reset_eq\",\"name\":\"Reset EQ\",\"type\":\"int\",\"min\":0,\"max\":1,\"step\":1},"
            "{\"key\":\"kit\",\"name\":\"Kit\",\"type\":\"int\",\"min\":0,\"max\":63,\"step\":1},"
            "{\"key\":\"rnd_kit\",\"name\":\"Rnd Kit\",\"type\":\"int\",\"min\":0,\"max\":1,\"step\":1},"
            "{\"key\":\"rnd_voice\",\"name\":\"Rnd Voice\",\"type\":\"int\",\"min\":0,\"max\":1,\"step\":1},"
            "{\"key\":\"rnd_pitch\",\"name\":\"Rnd Pitch\",\"type\":\"int\",\"min\":0,\"max\":1,\"step\":1},"
            "{\"key\":\"init_freq\",\"name\":\"Init Freq\",\"type\":\"int\",\"min\":0,\"max\":1,\"step\":1},"
            "{\"key\":\"rnd_pan\",\"name\":\"Rnd Pan\",\"type\":\"int\",\"min\":0,\"max\":1,\"step\":1},"
            "{\"key\":\"all_mono\",\"name\":\"All Mono\",\"type\":\"int\",\"min\":0,\"max\":1,\"step\":1},"
            "{\"key\":\"save_kit\",\"name\":\"Save Kit\",\"type\":\"enum\",\"options\":[\"Play\",\"Save\"]},"
            "{\"key\":\"v1_pan\",\"name\":\"V1 Pan\",\"type\":\"float\",\"min\":-1,\"max\":1,\"step\":0.02},"
            "{\"key\":\"v2_pan\",\"name\":\"V2 Pan\",\"type\":\"float\",\"min\":-1,\"max\":1,\"step\":0.02},"
            "{\"key\":\"v3_pan\",\"name\":\"V3 Pan\",\"type\":\"float\",\"min\":-1,\"max\":1,\"step\":0.02},"
            "{\"key\":\"v4_pan\",\"name\":\"V4 Pan\",\"type\":\"float\",\"min\":-1,\"max\":1,\"step\":0.02},"
            "{\"key\":\"v5_pan\",\"name\":\"V5 Pan\",\"type\":\"float\",\"min\":-1,\"max\":1,\"step\":0.02},"
            "{\"key\":\"v6_pan\",\"name\":\"V6 Pan\",\"type\":\"float\",\"min\":-1,\"max\":1,\"step\":0.02},"
            "{\"key\":\"v7_pan\",\"name\":\"V7 Pan\",\"type\":\"float\",\"min\":-1,\"max\":1,\"step\":0.02},"
            "{\"key\":\"v8_pan\",\"name\":\"V8 Pan\",\"type\":\"float\",\"min\":-1,\"max\":1,\"step\":0.02},"
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
            "{\"key\":\"v8_level\",\"name\":\"V8 Level\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"cv_vol\",\"name\":\"Volume\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"cv_pan\",\"name\":\"Pan\",\"type\":\"float\",\"min\":-1,\"max\":1,\"step\":0.02},"
            "{\"key\":\"cv_freq\",\"name\":\"Freq\",\"type\":\"int\",\"min\":20,\"max\":20000,\"step\":1},"
            "{\"key\":\"cv_attack\",\"name\":\"Attack\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.001},"
            "{\"key\":\"cv_decay\",\"name\":\"Decay\",\"type\":\"float\",\"min\":0,\"max\":4,\"step\":0.01},"
            "{\"key\":\"cv_wave\",\"name\":\"Wave\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"cv_penv\",\"name\":\"P.Env\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"cv_prate\",\"name\":\"P.Rate\",\"type\":\"float\",\"min\":0,\"max\":2,\"step\":0.01},"
            "{\"key\":\"cv_lamt\",\"name\":\"LFO Amt\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"cv_lrate\",\"name\":\"LFO Rate\",\"type\":\"float\",\"min\":0.1,\"max\":80,\"step\":0.5},"
            "{\"key\":\"cv_ftype\",\"name\":\"Flt Type\",\"type\":\"enum\",\"options\":[\"LP\",\"HP\",\"BP\"]},"
            "{\"key\":\"cv_cutoff\",\"name\":\"Cutoff\",\"type\":\"int\",\"min\":20,\"max\":18000,\"step\":1},"
            "{\"key\":\"cv_fres\",\"name\":\"Reso\",\"type\":\"float\",\"min\":1,\"max\":5,\"step\":0.1},"
            "{\"key\":\"cv_nattack\",\"name\":\"N.Attack\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.001},"
            "{\"key\":\"cv_ndecay\",\"name\":\"N.Decay\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"cv_mix\",\"name\":\"Mix\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"cv_dist\",\"name\":\"Distort\",\"type\":\"float\",\"min\":0,\"max\":50,\"step\":0.5},"
            "{\"key\":\"cv_level\",\"name\":\"Level\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"cv_preset\",\"name\":\"Preset\",\"type\":\"int\",\"min\":0,\"max\":40,\"step\":1},"
            "{\"key\":\"cv_rsend\",\"name\":\"Rev Send\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"cv_dsend\",\"name\":\"Dly Send\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"rev_mix\",\"name\":\"Rev Mix\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"rev_type\",\"name\":\"Rev Type\",\"type\":\"enum\",\"options\":[\"Club\",\"Garage\",\"Studio\"]},"
            "{\"key\":\"rev_size\",\"name\":\"Rev Size\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"rev_decay\",\"name\":\"Rev Decay\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"dly_mix\",\"name\":\"Dly Mix\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"dly_rate\",\"name\":\"Dly Rate\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"dly_fdbk\",\"name\":\"Dly Fdbk\",\"type\":\"float\",\"min\":0,\"max\":0.95,\"step\":0.01},"
            "{\"key\":\"dly_tone\",\"name\":\"Dly Tone\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v1_rsend\",\"name\":\"V1 Rev Send\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v1_dsend\",\"name\":\"V1 Dly Send\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v2_rsend\",\"name\":\"V2 Rev Send\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v2_dsend\",\"name\":\"V2 Dly Send\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v3_rsend\",\"name\":\"V3 Rev Send\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v3_dsend\",\"name\":\"V3 Dly Send\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v4_rsend\",\"name\":\"V4 Rev Send\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v4_dsend\",\"name\":\"V4 Dly Send\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v5_rsend\",\"name\":\"V5 Rev Send\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v5_dsend\",\"name\":\"V5 Dly Send\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v6_rsend\",\"name\":\"V6 Rev Send\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v6_dsend\",\"name\":\"V6 Dly Send\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v7_rsend\",\"name\":\"V7 Rev Send\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v7_dsend\",\"name\":\"V7 Dly Send\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v8_rsend\",\"name\":\"V8 Rev Send\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"v8_dsend\",\"name\":\"V8 Dly Send\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01}"
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
            "\"root\":{\"name\":\"Weird Dreams\","
            "\"knobs\":[\"cv_vol\",\"cv_pan\",\"cv_freq\",\"cv_decay\",\"cv_wave\",\"cv_mix\",\"cv_cutoff\",\"cv_preset\"],"
            "\"params\":[{\"level\":\"Patch\",\"label\":\"Patch\"},{\"level\":\"General\",\"label\":\"General\"},{\"level\":\"Voice\",\"label\":\"Voice\"},{\"level\":\"FX\",\"label\":\"FX\"}]},"
            "\"Patch\":{\"label\":\"Patch\","
            "\"knobs\":[\"kit\",\"rnd_kit\",\"rnd_voice\",\"rnd_pitch\",\"same_freq\",\"init_freq\",\"rnd_pan\",\"all_mono\"],"
            "\"params\":[\"kit\",\"rnd_kit\",\"rnd_voice\",\"rnd_pitch\",\"same_freq\",\"init_freq\",\"rnd_pan\",\"all_mono\",\"save_kit\"]},"
            "\"General\":{\"label\":\"General\","
            "\"knobs\":[\"comp\",\"dj_filter\",\"eq_lo\",\"lo_freq\",\"eq_mid\",\"mid_freq\",\"eq_hi\",\"hi_freq\"],"
            "\"params\":[\"comp\",\"dj_filter\",\"eq_lo\",\"lo_freq\",\"eq_mid\",\"mid_freq\",\"eq_hi\",\"hi_freq\",\"q_lo\",\"q_mid\",\"q_hi\",\"reset_eq\",\"master\"]},"
            "\"Voice\":{\"label\":\"Voice\","
            "\"knobs\":[\"cv_vol\",\"cv_pan\",\"cv_freq\",\"cv_decay\",\"cv_wave\",\"cv_mix\",\"cv_cutoff\",\"cv_preset\"],"
            "\"params\":[\"cv_vol\",\"cv_pan\",\"cv_freq\",\"cv_attack\",\"cv_decay\",\"cv_wave\",\"cv_penv\",\"cv_prate\",\"cv_lamt\",\"cv_lrate\",\"cv_ftype\",\"cv_cutoff\",\"cv_fres\",\"cv_nattack\",\"cv_ndecay\",\"cv_mix\",\"cv_dist\",\"cv_level\",\"cv_preset\",\"cv_rsend\",\"cv_dsend\"]},"
            "\"FX\":{\"label\":\"FX\","
            "\"knobs\":[\"rev_mix\",\"rev_type\",\"rev_size\",\"rev_decay\",\"dly_mix\",\"dly_rate\",\"dly_fdbk\",\"dly_tone\"],"
            "\"params\":[\"rev_mix\",\"rev_type\",\"rev_size\",\"rev_decay\",\"dly_mix\",\"dly_rate\",\"dly_fdbk\",\"dly_tone\"]}"
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
        /* FX */
        n += snprintf(buf + n, buf_len - n, "%.4f %d %.4f %.4f %.4f %.4f %.4f %.4f ",
            inst->reverb.mix, inst->reverb.type, inst->reverb.size, inst->reverb.decay,
            inst->delay.mix, inst->delay.rate, inst->delay.feedback, inst->delay.tone);
        /* Per-voice */
        for (int i = 0; i < NUM_VOICES; i++) {
            wd_voice_t *v = &inst->voice[i];
            n += snprintf(buf + n, buf_len - n,
                "%d %.1f %.4f %.4f %.4f %.3f %.4f %.3f %.1f %d %.0f %.2f %.4f %.4f %.3f %.1f %.3f %.2f %.4f %.4f ",
                v->preset, v->freq, v->attack, v->decay, v->wave,
                v->pitch_env_amt, v->pitch_env_rate, v->pitch_lfo_amt, v->pitch_lfo_rate,
                v->filter_type, v->filter_cutoff, v->filter_res,
                v->noise_attack, v->noise_decay, v->mix, v->distortion, v->level, v->pan,
                v->reverb_send, v->delay_send);
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
    if (strcmp(key, "rnd_kit") == 0) return snprintf(buf, buf_len, "0");
    if (strcmp(key, "rnd_voice") == 0) return snprintf(buf, buf_len, "0");
    if (strcmp(key, "rnd_pitch") == 0) return snprintf(buf, buf_len, "0");
    if (strcmp(key, "rnd_pan") == 0) return snprintf(buf, buf_len, "0");
    if (strcmp(key, "init_freq") == 0) return snprintf(buf, buf_len, "0");
    if (strcmp(key, "all_mono") == 0) return snprintf(buf, buf_len, "0");
    if (strcmp(key, "save_kit") == 0) return snprintf(buf, buf_len, "%s", inst->save_kit_state ? "Save" : "Play");
    if (strcmp(key, "reset_eq") == 0) return snprintf(buf, buf_len, "0");
    if (strcmp(key, "same_freq") == 0) return snprintf(buf, buf_len, "%d", inst->same_freq > 0 ? (int)inst->same_freq : 0);
    if (strcmp(key, "master") == 0) return snprintf(buf, buf_len, "%.4f", inst->master.master_level);

    /* FX params */
    if (strcmp(key, "rev_mix") == 0) return snprintf(buf, buf_len, "%.4f", inst->reverb.mix);
    if (strcmp(key, "rev_type") == 0) return snprintf(buf, buf_len, "%s", REV_TYPE_NAMES[inst->reverb.type]);
    if (strcmp(key, "rev_size") == 0) return snprintf(buf, buf_len, "%.4f", inst->reverb.size);
    if (strcmp(key, "rev_decay") == 0) return snprintf(buf, buf_len, "%.4f", inst->reverb.decay);
    if (strcmp(key, "dly_mix") == 0) return snprintf(buf, buf_len, "%.4f", inst->delay.mix);
    if (strcmp(key, "dly_rate") == 0) return snprintf(buf, buf_len, "%.4f", inst->delay.rate);
    if (strcmp(key, "dly_fdbk") == 0) return snprintf(buf, buf_len, "%.4f", inst->delay.feedback);
    if (strcmp(key, "dly_tone") == 0) return snprintf(buf, buf_len, "%.4f", inst->delay.tone);

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
        snprintf(k, sizeof(k), "v%d_rsend", i+1);
        if (strcmp(key, k) == 0) return snprintf(buf, buf_len, "%.4f", v->reverb_send);
        snprintf(k, sizeof(k), "v%d_dsend", i+1);
        if (strcmp(key, k) == 0) return snprintf(buf, buf_len, "%.4f", v->delay_send);
    }

    /* Virtual cv_* params — redirect to current voice */
    if (strncmp(key, "cv_", 3) == 0) {
        int vi = inst->current_voice;
        wd_voice_t *v = &inst->voice[vi];
        const char *suffix = key + 3;
        if (strcmp(suffix, "vol") == 0) return snprintf(buf, buf_len, "%.4f", inst->voice_vol[vi]);
        if (strcmp(suffix, "pan") == 0) return snprintf(buf, buf_len, "%.4f", v->pan);
        if (strcmp(suffix, "freq") == 0) return snprintf(buf, buf_len, "%d", (int)v->freq);
        if (strcmp(suffix, "decay") == 0) return snprintf(buf, buf_len, "%.4f", v->decay);
        if (strcmp(suffix, "wave") == 0) return snprintf(buf, buf_len, "%.4f", v->wave);
        if (strcmp(suffix, "mix") == 0) return snprintf(buf, buf_len, "%.4f", v->mix);
        if (strcmp(suffix, "cutoff") == 0) return snprintf(buf, buf_len, "%d", (int)v->filter_cutoff);
        if (strcmp(suffix, "preset") == 0) return snprintf(buf, buf_len, "%d", v->preset);
        if (strcmp(suffix, "attack") == 0) return snprintf(buf, buf_len, "%.4f", v->attack);
        if (strcmp(suffix, "penv") == 0) return snprintf(buf, buf_len, "%.4f", v->pitch_env_amt);
        if (strcmp(suffix, "prate") == 0) return snprintf(buf, buf_len, "%.4f", v->pitch_env_rate);
        if (strcmp(suffix, "lamt") == 0) return snprintf(buf, buf_len, "%.4f", v->pitch_lfo_amt);
        if (strcmp(suffix, "lrate") == 0) return snprintf(buf, buf_len, "%.4f", v->pitch_lfo_rate);
        if (strcmp(suffix, "ftype") == 0) return snprintf(buf, buf_len, "%s", FILTER_NAMES[v->filter_type]);
        if (strcmp(suffix, "fres") == 0) return snprintf(buf, buf_len, "%.4f", v->filter_res);
        if (strcmp(suffix, "nattack") == 0) return snprintf(buf, buf_len, "%.4f", v->noise_attack);
        if (strcmp(suffix, "ndecay") == 0) return snprintf(buf, buf_len, "%.4f", v->noise_decay);
        if (strcmp(suffix, "dist") == 0) return snprintf(buf, buf_len, "%.4f", v->distortion);
        if (strcmp(suffix, "level") == 0) return snprintf(buf, buf_len, "%.4f", v->level);
        if (strcmp(suffix, "rsend") == 0) return snprintf(buf, buf_len, "%.4f", v->reverb_send);
        if (strcmp(suffix, "dsend") == 0) return snprintf(buf, buf_len, "%.4f", v->delay_send);
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
        float rev_in = 0.0f, dly_in = 0.0f;

        for (int v = 0; v < NUM_VOICES; v++) {
            float vol = onepole(&inst->voice_vol_smooth[v], inst->voice_vol[v], 0.002f);
            float pan = onepole(&inst->voice_pan_smooth[v], inst->voice[v].pan, 0.002f);
            float sample = voice_render_sample(&inst->voice[v]) * vol;
            /* Constant-power panning */
            float angle = (pan + 1.0f) * 0.25f * 3.14159265f;
            mix_l += sample * cosf(angle);
            mix_r += sample * sinf(angle);
            /* Send buses (pre-pan mono) */
            rev_in += sample * inst->voice[v].reverb_send;
            dly_in += sample * inst->voice[v].delay_send;
        }

        /* Scale down (8 voices) */
        mix_l *= 0.35f;
        mix_r *= 0.35f;
        rev_in *= 0.35f;
        dly_in *= 0.35f;

        /* Process stereo reverb and delay */
        float rev_l, rev_r, dly_l, dly_r;
        reverb_process_stereo(&inst->reverb, rev_in, &rev_l, &rev_r);
        delay_process_stereo(&inst->delay, dly_in, &dly_l, &dly_r);

        /* Mix FX returns into stereo bus */
        mix_l += rev_l * inst->reverb.mix + dly_l * inst->delay.mix;
        mix_r += rev_r * inst->reverb.mix + dly_r * inst->delay.mix;

        /* Master FX (process L and R independently) */
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
