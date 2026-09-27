# root.tex Review (v3, full re-pass)

Fresh full pass over the current `root.tex` and its `\input`s, since the
previous `ROOT_REVIEW.md` is now stale (several items fixed, some regressed,
some new). Where something from the last review was fixed, it's marked
✅ and not repeated in detail. Line numbers are current as of this pass.

## 0. Resolved since last review (good — confirmed, not re-flagging)

- ✅ `color-edits` is now loaded with `[suppress]` (`preamble.tex:149`), so
  the review-comment macros no longer render as visible colored text.
  Verified the old plain `\usepackage{color-edits}` above it is commented out.
- ✅ Introduction contributions paragraph (L117) still reads cleanly.
- ✅ The "sufficient for sufficient systems" duplicated-word bug at L129 is
  fixed (now "sensitive systems").
- ✅ Dangling "...as a nonlinear change of coordinates on $C$, with the new
  parameterization" (old L248) is fixed — now ends cleanly at "...on $C$."
- ✅ PDF-size fix applied and images swapped in (`realmazex.jpg`,
  `hero_maze2.jpg` now referenced instead of the oversized PNGs).
- ✅ `hyperref` no longer forces `allcolors=purduegold` (reverted to default
  colorlinks palette, per your last request).
- ✅ `ruby_results.tex` table no longer bleeds off the column edge.

## 1. New/regressed: a real factual inconsistency (DoF count)

The paper now states **three different DoF counts** for the same RB-Y1 robot
across three places:

- **L50** (Introduction): *"a **23-DoF** bimanual mobile manipulator"*
- **L75** (Fig. 1 caption): *"A **20-DoF** bimanual mobile manipulator..."*
- **L647** (§ Pick and Place): *"...which makes it a **20-DoF** system."*

20-DoF is used everywhere else (abstract, L117, Fig. 1 caption, the actual
experiment section), so L50 is very likely a stray leftover from an earlier
draft (there's also a dead commented-out `\subsubsection{23-DoF Humanoid...}`
at L401, suggesting 23-DoF was an earlier, since-superseded design point).
**Fix L50 to 20-DoF** — this is the kind of inconsistency a reviewer flags
immediately since it's visible within the first paragraph and the very next
figure caption contradicts it.

## 2. New regression: dangling clause at L245

> Specifying a discrete branch $b$ and, a continuous self-motion parameter
> $\psi \in \Psi$ gives a single-valued map...

This used to read "...a discrete branch $b$ and, **for redundant arms**, a
continuous self-motion parameter..." — that qualifier got dropped somewhere
along the way, leaving a stray comma with nothing between "and," and "a
continuous." Either restore "for redundant arms" (it's semantically load-
bearing — non-redundant arms don't have a free parameter) or just remove the
comma: "Specifying a discrete branch $b$ and a continuous self-motion
parameter $\psi \in \Psi$..."

## 3. `\monogram` macro used inconsistently

You defined `\newcommand{\monogram}[3]{{}^{#2}\!#1^{#3}}` and use it at L566
(`\monogram{X}{l}{r}`) and L578 (`\monogram{X}{W}{B}`), but the very same
kind of quantity reverts to hand-written `{}^{B}X^{l}`, `{}^{W}X^{B}` etc.
**in the rest of the same paragraph** (L578 continues with
`${}^{B}X^{l}$`/`${}^{B}X^{r}$`/`${}^{W}X^{B}\,{}^{B}X^{l}$`), and again at
L580, L657, L569. If the macro is worth introducing, it's worth using
consistently for every left/right-superscripted transform in the paper —
right now a reader sees both notations for the identical construct within
one paragraph, which reads as an oversight rather than a style choice.

Minor: L566 also has a stray space before the period —
`\FK(q_r) = \monogram{X}{l}{r}$ .` (space between `$` and `.`).

## 4. Still-open grammar/typos from the last pass (unchanged, listed once, concise)

- **L440** `on an 5.4GHz Intel i7-13700K CPU` → `a 5.4GHz`; also inconsistent
  spacing vs. `2.5~GHz` (L539) — should be `5.4~GHz`.
- **L647** `Our final experiment deploys our the planner` — duplicated word.
- **L647** `the parameterization based on the Inverse Function Theorem (IFT)
  work proposed by~\cite{cohn2026planning}` — still a dangling appositive
  with no verb connecting it to the sentence; needs "..., **using** a
  parameterization based on...".
- **L647** `With a slight abuse in terminology` — idiom is "abuse **of**
  terminology."
- **L673** item (2) `unconstrained plan from start-to-pre-pick configuration,
  and place-to-home configuration` — still breaks parallel structure with
  the other list items.
- **L695** — still a three-clause run-on/comma splice: *"Although McVAMP has
  slightly lower atomic planning latency for successful plans, it takes
  longer to fail, which would be exacerbated by the large number of planning
  calls we make, the complete ReVAMP pipeline is faster because..."* Needs
  restructuring (e.g., break after "planning calls we make" into its own
  clause with "so").
- **L710** `Recent works has shown` — subject/verb disagreement, should be
  "Recent **work** has shown" (matches "work" used singularly elsewhere,
  e.g. L57 "Recent advances").
- **L42 (abstract)** `reparameterizating` → `reparameterizing`; `such
  planning speeds opens up avenues` → `speeds **open** up`; abstract's bare
  `10x` still inconsistent with body's `$10\times$` notation.
- **L129** comma splice *"...is often an expensive operation, they can
  satisfy..."* and `upto` → `up to` (also at L355, L665 `3cm` → `3~cm`).
- **L368** `unlike joint-space samples, that requires forward-kinematics` →
  "which **require**"; and `This eliminates a significant number of IK calls
  and further denser collision checking in cluttered environments` is still
  missing a verb in its second half.
- **L284** `units that **maps**` → `units that **map**`.
- **L289** `we carefully modify in two stages` — still missing a direct
  object (modify *what*?).
- **L301** `Euclidian` (×2) → `Euclidean`.

## 5. `et al.` is now inconsistent in a new way

L566 was fixed to `Cohn et al.` (correct), but the exact same author is
still `Cohn et. al` at **L569** and **L629**, and `He. et. al` (double error
— stray period after "He" *and* after "et") at **L486**. Since one instance
is already correctly fixed, this now reads as an inconsistency rather than a
uniform style choice — worth a single find-and-replace pass:
`et\. al\.?` → `et al.`, and fix `He\. et\. al` → `He et al.` separately.

## 6. Table column width: `ruby_results.tex` may have regressed

The label column was tuned to `p{2.3cm}` (verified against a mockup at your
actual column width/font to just barely avoid overflow). The file on disk
now has **`p{3.3cm}`** — a full centimeter wider — and the row labels were
reverted to the original 4 separate `mean (ms)`/`max (ms)` rows (that's your
call, not flagging it as wrong), but note:

- At `p{3.3cm}`, the 3 data columns each have noticeably less room than what
  I verified fits; this may reintroduce the right-edge bleed the whole
  exercise was trying to fix. Worth a recompile check.
- **L136**: `Unsucc.\ Call Time, max(ms)` — still missing the space before
  `(ms)` (every other row has `, max (ms)` with a space; this one doesn't).

## 7. Orphan figures — still unreferenced (unchanged from last review)

`\label{tab:maze_results}` (L514, the maze CDF/box-plot figure) and
`\label{fig:bimanual_results}` (L622, the bimanual CDF/box-plot figure) are
still never cited anywhere via `\cref`/`\ref` — only their companion
*tables* (`tab:sim_maze_results`, `tab:bimanual_results`) get referenced in
the running text. Confirmed via grep this pass; still open.

## 8. Figure/subfigure labels still prefixed `tab:` (unchanged)

`tab:maze_time`, `tab:maze_distance`, `tab:maze_results`, `tab:bimanual_time`,
`tab:bimanual_distance` are all figure/subfigure labels, not tables. Still
worth renaming to `fig:...` to avoid confusion with the real
`tab:sim_maze_results`/`tab:bimanual_results` tables. Unchanged from last
review.

## 9. Stale review comments that are already resolved (cleanup, not urgent)

Since `[suppress]` is on, these no longer print, but they're still sitting
in the source as if unresolved, which will confuse the next person reading
raw `.tex`:

- **L691** `\tccomment{Why "retraction"? Maybe "waypoint"?}` — already
  fixed; the text now says "waypoint" (both here and at L673). Delete the
  comment.
- **L620** `\tccomment{Make sure to be consistent on the terminology for two
  followers (used "Task-Space IK" in previous section)}` — "Task-Space IK"
  no longer appears anywhere in the live text (superseded by
  "Leader-Follower"/"Dual-Follower" throughout); this comment is stale.

Still genuinely open (worth actually resolving, not just deleting):

- **L301** `\tccomment{would we cite something here?}` /
  `\sicomment{its from cephes which is ancient and industry standard}` — if
  Cephes is the actual source, this reads like it just needs a citation
  added to the sentence and then both comments deleted.
- **L566** `\tccomment{Might need to cite Russ' textbook for monogram
  notation}` — still no citation added; either add one or drop the comment
  if you've decided against it.
- **L521** `\tccomment{Don't think you've discussed this yet?}` — worth
  checking whether the "$(x,y)$ start-goal queries vs. constraint-satisfying
  configurations" distinction is set up earlier in the paper before this
  point; if not, this is a real content gap, not just a stale note.

## 10. Unchanged since last review, still open

- `\input{crrtc_algo.tex}` (L435) still commented out; file still unused.
- The fully-commented `Parameterization Space for different robots`
  subsection (`\label{sec:arms}`, ~L378-432) and its dead summary table
  (`\label{tab:param_summary}`, ~L442-456) are still there, unreferenced.
- `tables/maze/real_results_table.tex` still has ~20 lines of commented-out
  duplicate tables at its top sharing labels with live content elsewhere
  (inert, but a silent-duplicate-label trap if uncommented by accident).

## Suggested order of fixes

1. **L50 DoF fix** (23→20) — highest-visibility factual inconsistency, one
   character change.
2. **L245** dangling clause, **L566** stray space + `\monogram` consistency
   sweep — these touch the paper's core formalism and are worth getting
   right together.
3. `et al.` sweep (§5) and the mechanical typo list (§4) — can be done in
   one pass.
4. Recompile and eyeball `ruby_results.tex` for right-edge bleed given the
   widened label column (§6), and fix the one remaining `max(ms)` spacing typo.
5. Decide on the orphan-figure citations (§7) and stale-comment cleanup (§9).
