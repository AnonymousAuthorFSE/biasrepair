# Manual evaluation guidelines

Structured manual evaluation of BiasRepair rewrites: 100 sentences total, 20 per bias category (Generic Pronouns, Exclusionary Terms, Stereotyping Bias, Sexism, Semantic Bias).

## Sampling

- Randomly sample 100 rewritten sentences from the evaluation set.
- Stratify by category: 20 sentences from each of the five repair categories.
- Annotators are **blind** to the system configuration that produced each rewrite.

## Annotators

- Two independent annotators with background in software engineering and prior exposure to bias and inclusive language.
- In case of disagreement, a third independent annotator adjudicates through discussion.

## Criteria (4 dimensions, 5-point Likert scale 1–5)

1. **Bias Removal**: Was the targeted gender bias fully eliminated?
2. **Semantic Preservation**: Was the technical and pedagogical meaning of the original sentence retained?
3. **Fluency**: Grammatical correctness and naturalness of the rewrite.
4. **Pedagogical Fit**: Would the rewritten sentence be appropriate for use in software engineering course materials?

## Procedure

- Provide clear annotation guidelines with category-specific examples to ensure consistency.
- Each annotator evaluates each sampled rewrite along the four criteria.
- Record ratings in the rating sheet (one row per sentence; columns for criterion scores and annotator ID).
- Compute inter-rater agreement (Cohen’s kappa) on the initial two-annotator ratings before adjudication. Target: strong agreement (e.g. κ ≈ 0.81 as in the paper).
- Resolve disagreements via third annotator and discussion; final scores used for reporting.

## Output

- Average score per criterion across the 100 rewrites (as in paper Table: Bias Removal ~4.6, Semantic Preservation ~4.7, Fluency ~4.8, Pedagogical Fit ~4.5).
- Report Cohen’s kappa for inter-rater agreement.
