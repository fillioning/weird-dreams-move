# Weird Dreams — 8-Voice Drum Machine for Ableton Move

Port of [WeirdDrums](https://github.com/dfilaretti/WeirdDrums) (Daniele Filaretti, MIT) to the [Schwung](https://github.com/charlesvestal/move-everything) framework, expanded into a full 8-voice drum machine with master bus FX, 64 kit presets and randomization (kit, pitch, pan).

## Features

- **8 independent drum voices**, each with oscillator (continuous sine/tri/saw/square morph), white noise with SVF filter, exponential AD envelopes, pitch modulation, tanh distortion, and clap retrigger
- **Dynamic pad-selected voice editing** — press a pad to instantly switch knobs and menu to that voice's parameters (Mr Drums-style)
- **41 voice presets** across 8 categories: Kicks, Snares, Toms, Hi-Hats, Cymbals, Claps, Percussion, FX
- **64 kit presets**: classic machines (808, 909, LinnDrum, CR-78), genre kits (Techno, House, Trap, DnB, Afrobeat, Footwork...), character kits (Industrial, Lo-Fi, Ambient, Glitch...), and hybrid/specialized kits
- **96 musical pitch scales** (Rnd Pitch): 8 scale types (Major, Minor, Pentatonic, Blues, Modes, Harmonic/Melodic minor, Exotic) x 12 root notes, with per-voice frequency ranges tuned to real hardware drum synth standards
- **Master bus FX**: dirty compressor, Isolator3-style DJ filter, Massenburg 8200-inspired 3-band parametric EQ
- **Per-voice volume and panning** with constant-power law

## UI Pages

| Page | Knobs | Description |
|------|-------|-------------|
| **Patch** | Kit, Rnd Kit, Rnd Voice, Rnd Pitch, SameFreq, Init Freq, Rnd Pan, All Mono | Kit selection and randomization |
| **General** | Crush, Filter, Lo Gain, Lo Freq, Mid Gain, Mid Freq, Hi Gain, Hi Freq | Master bus FX (+ Q, Reset EQ, Master in menu) |
| **Voice** | Volume, Pan, Freq, Decay, Wave, Mix, Cutoff, Preset | Dynamic — edits the current pad's voice |

Pressing a pad triggers the voice AND switches to that voice's parameters on knobs and menu. The Voice page title shows which voice is selected.

## MIDI Mapping

- Notes C2-G#2 (36-43): trigger voices 1-8 directly
- Other notes: round-robin across all 8 voices

## Build

```bash
./scripts/build.sh          # Docker ARM64 cross-compile
./scripts/install.sh        # SCP to Move
```

## Credits

- Original DSP: [Daniele Filaretti](https://github.com/dfilaretti/WeirdDrums) (MIT)
- Move port and expansion: Vincent Fillion
- Framework: [Schwung](https://github.com/charlesvestal/move-everything)

## License

MIT
