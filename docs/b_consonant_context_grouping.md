# B Consonant Context Grouping

This document groups the current B-word list by the acoustic context immediately after the initial B cue. The goal is to build a context-aware B stub library instead of using one universal B stub for every word.

## Why This Matters

The initial /b/ is not acoustically identical across all words. The following vowel changes the release, voicing onset, and formant transition. A single B stub can work well for one word, such as `bird`, but sound weak or unnatural for another word, such as `base`.

For this project, the useful control target is not full word recognition. It is local context selection:

```text
detect B cue
-> estimate early vowel/transition context
-> choose B_front / B_central / B_back / B_r_colored / B_L_cluster
-> apply stub or additive enhancement
```

## Main Groups

| Group | Example Words | Why Separate |
|---|---|---|
| `B_front_high` | beach, bead, beat | High front vowel transition after B. |
| `B_front_high_lax` | bid, big, bin, bit | Lax high-front transition. |
| `B_front_mid` | base, bait, bay, bed, best | Front or front-mid transition. |
| `B_front_low` | back, bad, bag, ban, bash, bat | Low-front transition. |
| `B_central` | bud, bug, bus | Central vowel transition. |
| `B_back_low` | bob, bond, box | Low/back transition. |
| `B_back_rounded` | ball, boss | Rounded back vowel; accent dependent. |
| `B_back_diphthong` | boat, bone, both, bound | Rounded/back diphthong transition. |
| `B_back_high` | boot | High back rounded transition. |
| `B_back_high_lax` | book, bull | Lax back/high transition. |
| `B_r_colored` | bar, barn, burn | R-colored transition; should be tested separately. |
| `B_open_diphthong` | bite, buy | AY context; starts open and moves high. |
| `B_rounded_diphthong` | boy | OY context; rounded diphthong. |
| `B_L_cluster` | black, blade, blank, blast, blot, blue | Initial /bl/ cluster; ordinary B-only replacement can damage the L transition. |

## Metadata File

The row-level grouping is saved at:

```text
data/metadata/b_context_groups.csv
```

The most important columns are:

```text
word
onset_type
first_vowel_arpabet
vowel_context
suggested_stub_group
stub_priority
notes
```

## Product Interpretation

For the next phone-side prototype, start with these practical groups:

```text
B_front = front_high + front_high_lax + front_mid + front_low
B_central = central
B_back = back_low + back_rounded + back_diphthong + back_high + back_high_lax
B_r_colored = r_colored
B_L_cluster = bl_cluster
```

Then run A/B listening tests to decide whether we need fine groups or whether the coarse five-group version is enough.
