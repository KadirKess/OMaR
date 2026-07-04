# Musical Information Reconstruction — Implementation Guide

> Personal implementation guide for the *reconstruction* stage of the OMR pipeline
> (turning detected primitives into structured musical information).
> Each step references the research paper that justifies it. Quotes are taken from
> the papers' abstracts / summaries and are marked as such; everything else is a
> paraphrase with a link so you can read the source yourself.

---

## 0. Where this fits in the pipeline

Every OMR survey splits the problem into 4 stages. Your task is **stage 3**:

1. Preprocessing — *(done: binarization, deskew, staff handling)*
2. Symbol / primitive detection — *(done: notehead / stem / beam segmentation)*
3. **Notation assembly / reconstruction** ← **this guide**
4. Encoding to output (MusicXML / MEI / MIDI)

The single most important idea, from **Calvo-Zaragoza, Hajič & Pacha,
*Understanding Optical Music Recognition* (2020)** — split stage 3 into two
sub-problems and never mix them:

- **Syntactic reconstruction** — group primitives into symbols and recover their
  *spatial relationships* (this stem belongs to that notehead; these noteheads
  share this beam). Inherently **2D**, not linear.
- **Semantic interpretation** — assign *meaning*: pitch, duration, then voices,
  measures, timing.

> The paper notes that reconstruction of musical semantics is *often omitted from
> published articles because the operations are specific to the output format* —
> which is exactly why this stage is under-documented and why you have to build it
> carefully.

📄 Free PDF: https://arxiv.org/pdf/1908.03608 — read §2 (pipeline) and the
"notation assembly" discussion first.

---

## 1. Architecture decision — keep the engine format-independent

From **Bainbridge & Bell, *A music notation construction engine for optical music
recognition*, Software: Practice & Experience (2003)**: build the reconstruction
engine as a standalone module whose output is a **graph**, not MusicXML. Adding
MIDI/MEI later then becomes a pure serialization swap.

```
primitives ─▶ [reconstruction engine] ─▶ notation graph ─▶ [encoder] ─▶ MusicXML / MEI / MIDI
```

🔗 https://www.semanticscholar.org/paper/A-music-notation-construction-engine-for-optical-Bainbridge-Bell/fcda1e784373854a2dc84974b3dcb12ab6b639c2

- [ ] Decide the graph node/edge types (see §2) before writing any rule.
- [ ] Keep all output-format code out of the engine.

---

## 2. Data structure — the notation graph

From **Fahmy & Blostein, *A graph grammar programming style for recognition of
music notation*, Machine Vision and Applications 6(2):83–99 (1993)**. This is the
method closest to your setup: it *assumes a symbol recognizer already exists* (✓
your primitive splitting) and uses a graph grammar to assemble meaning from
**symbols + their spatial relationships**.

- **Nodes** = symbols. Reuse your `Primitive`; add higher-level node types as you
  rewrite (`Note`, `Chord`, `BeamedGroup`, `Measure`, …).
- **Edges** = typed spatial/syntactic relations. You already emit `head_to_stem`
  and `stem_to_beam` in `PrimitiveSplit` — generalize these into graph edges.
- **Attributes** on nodes = `pitch`, `duration`, `onset` (filled in during §4–§5).

🔗 https://link.springer.com/article/10.1007/BF01211933
📄 Free, closely related (graph-rewriting mechanics): https://research.cs.queensu.ca/home/blostein/LNCS1073.pdf

- [ ] Define `Node` and `Edge` types.
- [ ] Write an adapter from `Primitive` / `PrimitiveSplit` → graph nodes + seed edges.

---

## 3. Step 0 — coordinate frame (do this once)

Reference: *Understanding OMR* (2020), the syntactic/semantic split.

Convert every primitive from **pixels → (staff, diatonic-step, x-order)** using
your `staff_space` and `staff_line` (already carried in `PrimitiveSplit`).
Nothing downstream should touch pixels again.

- [ ] Recover the 5 staff-line y-positions per staff.
- [ ] Map each primitive bbox to a vertical staff position (in staff-space units).
- [ ] Assign a left-to-right x-order index within each staff.

---

## 4. Step 1 — BUILD the over-complete graph (Fahmy & Blostein)

Fahmy & Blostein's programming style is **Build–Weed–Incorporate**. This is the
BUILD phase: construct a deliberately *over-complete* graph.

- One node per primitive.
- Add a spatial edge between two primitives whenever a predicate holds, expressed
  in **staff-space units** (`touches`, `above`, `aligned_vertically`,
  `on_staff_position(n)`).
- **Ambiguity rule (their key trick):**

  > In their system, symbols can have ambiguous labels; *one node is created for
  > each possible value of the label*, allowing labelling errors (and stray dots)
  > to be corrected.

  So when a primitive is uncertain (low `confidence`, `SplitStatus.UNCERTAIN`),
  create **one candidate node per possible label** instead of committing. The
  WEED phase prunes the wrong ones. This is where your `confidence` field pays off.

- [ ] Implement the spatial predicates in staff-space units.
- [ ] Emit one node per primitive; multiple candidate nodes when uncertain.
- [ ] Emit all plausible spatial edges.

---

## 5. Step 2 — WEED + INCORPORATE (graph rewriting)

Apply rewriting rules that match a pattern and replace it with a single
higher-level node. Fahmy & Blostein describe this as replacing complex syntactic
patterns with simpler ones until the desired level of detail is distilled
(also formalised as *discrete relaxation* in their 1998 paper).

Rules (start small):

| Pattern | Rewrites to |
|---|---|
| `notehead` + touching `stem` | `note` |
| `notehead`(s) vertically aligned on one `stem` | `chord` |
| `note —beam— note …` | `beamed_group` |
| `… barline` | closes a `measure` |

Each rule **weeds** candidate nodes no rule can consume (the wrong labels from
§4) and **incorporates** the matched pattern into one node. Loop until no rule
fires. The result is your **syntactic reconstruction**.

> Note: Coüasnon's grammar productions (DCG) and Fahmy & Blostein's graph rules
> encode the *same* constraints in two notations. Pick whichever is easier to
> write in Python. Coüasnon ref:
> https://www.semanticscholar.org/paper/DMOS:-a-generic-document-recognition-method,-to-an-Co%C3%BCasnon/c8afc676c5f2ee8090b67b4c60ab2d8a9e3ebd4a

- [ ] Implement the rewrite loop (match → weed → incorporate → repeat).
- [ ] Start with `note`, then `chord`, then `beamed_group`, then `measure`.

---

## 6. Step 3 — semantic pass: DURATION (local rules)

Per `note` node, purely local — no ordering needed:

`duration = f(notehead fill (open/filled), stem present?, beam count / flag count, augmentation dots)`

- [ ] Map notehead fill + stem + beams/flags + dots → duration value.

---

## 7. Step 4 — semantic pass: PITCH (attribute inheritance)

This is what **attribute grammars** are for — the core of Coüasnon's approach
(**Coüasnon, Brisset & Stéphan, *Using logic programming languages for optical
music recognition*, 1995**, already in your `references.bib`). Traverse each
measure left-to-right carrying **inherited state**:

- current **clef** → maps a staff position to a pitch letter,
- **key signature**,
- **accidental persistence** — an accidental holds for the rest of that measure.

- [ ] Cut symbols into measures at barlines; order by x.
- [ ] Propagate clef / key / accidental state; compute each note's `pitch`.

---

## 8. Step 5 — VALIDATE + feedback

**Rossant & Bloch, *Robust and Adaptive OMR System Including Fuzzy Modeling,
Fusion of Musical Rules, and Possible Error Detection*, EURASIP JASP (2007)** —
use musical rules as a consistency check.

- **Constraint:** the sum of note durations in each measure must equal the time
  signature.
- **On failure:** you still hold competing candidate nodes from §4 — pick the
  combination that satisfies the constraint (graph-level discrete relaxation, the
  Fahmy & Blostein 1998 idea).
- **Optional, high value — Coüasnon's DMOS feedback loop:** the DMOS method
  (EPF grammar + parser) is designed to *inject context back into segmentation* —
  if no candidate satisfies the measure, re-query the **image** at the predicted
  location with a looser threshold ("a stem should be here"). Even a weak version
  measurably helps.

📄 Free full text (Rossant & Bloch): https://asp-eurasipjournals.springeropen.com/articles/10.1155/2007/81541
🔗 DMOS (Coüasnon): https://link.springer.com/article/10.1007/s10032-005-0148-5

- [ ] Implement the per-measure duration check.
- [ ] On violation, select the candidate combination that satisfies it.
- [ ] (Optional) add the image-feedback re-query.

---

## 9. Step 6 — ENCODE (pure serialization)

Per Bainbridge & Bell's separation principle: walk the validated graph →
MusicXML / MEI → MIDI. **Zero musical logic here.**

- [ ] Graph → MusicXML (or MEI).
- [ ] → MIDI (the "export MIDI" goal from the project README).

---

## Mapping to your existing code

| Your code (`prototype/src/primitives.py`) | Role in this guide |
|---|---|
| `Primitive` (kind, mask, confidence, bbox) | graph nodes; `confidence` drives §4 ambiguity |
| `head_to_stem`, `stem_to_beam` | seed edges for BUILD (§4) |
| `staff_space`, `staff_line` | Step 0 coordinate frame (§3) |
| `SplitStatus.UNCERTAIN` | triggers multi-candidate nodes (§4) |

---

## Suggested build order (milestones)

1. **Monophonic, single staff, treble clef.** Steps 0→4, skip chords/voices.
2. Add **measure + duration validation** (§8).
3. Add **chords**, then **beamed groups**.
4. Add **multi-voice** and the **Coüasnon image-feedback loop** last.

---

## Full reference list

**Grammar-based reconstruction (core methods)**
- Fahmy & Blostein, *A graph grammar programming style for recognition of music
  notation*, Machine Vision and Applications 6(2):83–99, 1993 —
  https://link.springer.com/article/10.1007/BF01211933
- Fahmy & Blostein, *A graph-rewriting paradigm for discrete relaxation:
  application to sheet-music recognition*, IJPRAI 12(6):763–799, 1998 (search on
  Google Scholar / Semantic Scholar).
- Blostein, *Computing with graphs and graph transformations*, SP&E 29(3), 1999 —
  https://onlinelibrary.wiley.com/doi/abs/10.1002/(SICI)1097-024X(199903)29:3%3C197::AID-SPE228%3E3.0.CO;2-F
- *Issues in the Practical Use of Graph Rewriting* (free PDF) —
  https://research.cs.queensu.ca/home/blostein/LNCS1073.pdf
- Coüasnon, Brisset & Stéphan, *Using logic programming languages for OMR*, 1995
  *(in your references.bib)*.
- Coüasnon, *DMOS: a generic document recognition method* —
  https://www.semanticscholar.org/paper/DMOS:-a-generic-document-recognition-method,-to-an-Co%C3%BCasnon/c8afc676c5f2ee8090b67b4c60ab2d8a9e3ebd4a
  and https://link.springer.com/article/10.1007/s10032-005-0148-5

**Architecture / reconstruction engine**
- Bainbridge & Bell, *A music notation construction engine for OMR*, SP&E, 2003 —
  https://www.semanticscholar.org/paper/A-music-notation-construction-engine-for-optical-Bainbridge-Bell/fcda1e784373854a2dc84974b3dcb12ab6b639c2

**Validation / error detection**
- Rossant & Bloch, *Robust and Adaptive OMR System…*, EURASIP JASP, 2007 (free) —
  https://asp-eurasipjournals.springeropen.com/articles/10.1155/2007/81541

**Framing / survey**
- Calvo-Zaragoza, Hajič & Pacha, *Understanding Optical Music Recognition*, 2020
  (free) — https://arxiv.org/pdf/1908.03608
- Blostein & Baird, *A Critical Survey of Music Image Analysis*, 1992
  *(in your references.bib)*.

---

## ⚠️ Honesty note on "chart parsers"

A *chart parser* (Earley/CYK) is a general parsing technique from CS/NLP, **not a
documented OMR method** — no OMR paper found uses one. Real grammar-based OMR uses
**DCG-with-backtracking** (Coüasnon) or **graph rewriting** (Fahmy & Blostein).
You may still use chart parsing as an implementation tactic for memoized ambiguity
handling, but present it as general parsing theory (Earley, 1970 —
https://user.phil.hhu.de/~kallmeyer/Parsing/earley.pdf), not as an OMR result.
