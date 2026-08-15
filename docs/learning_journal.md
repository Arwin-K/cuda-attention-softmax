# Learning journal

Use one entry for each major concept. Generated project notes may describe the
concept and point to code or evidence, but the student's prior understanding,
personal learning, surprises, and remaining questions must stay as
`TODO(student)` until the student writes them.

## Concept entry template

### Concept

`TODO: Name the concept.`

#### What I thought before

`TODO(student): Describe your prior mental model in your own words.`

#### What I learned

`TODO(student): Explain what changed in your understanding.`

#### Why it matters

`TODO: Explain the correctness, performance, or engineering relevance.`

#### Mental model

`TODO: Give a compact analogy, diagram, or step-by-step model.`

#### Where it appears in code

`TODO: Link the relevant file, function, and Git commit.`

#### Experiment demonstrating it

`TODO: Name the controlled experiment or state that no experiment exists yet.`

#### Evidence/result

`TODO: Link measured output or explicitly write "Not measured yet."`

#### Explain it in my own words

`TODO(student): Explain this concept without copying the generated notes.`

#### Questions still open

`TODO(student): List what you still want to understand or test.`

## Entry checklist

- Separate facts, measurements, and interpretations.
- Link measurements to raw artifacts and the implementation Git commit.
- Do not turn a hypothesis into an observed result.
- Do not write first-person reflection on the student's behalf.

## Stable softmax

### Concept

Maximum subtraction for numerically stable row-wise softmax.

#### What I thought before

`TODO(student): Describe your prior mental model in your own words.`

#### What I learned

`TODO(student): Explain what changed in your understanding.`

#### Why it matters

Direct FP32 exponentiation can overflow for large positive logits. Subtracting
the row maximum bounds the largest exponential at one without changing the
mathematical probability.

#### Mental model

Shift every competitor by the same amount before comparing them; their relative
differences remain unchanged, but the numerical range becomes safer.

#### Where it appears in code

`cuda_attention/reference.py::stable_softmax`, introduced in Commit 006.

#### Experiment demonstrating it

`tests/test_numerical_stability.py` compares large inputs with PyTorch and
demonstrates overflow in a deliberately naïve exponential.

#### Evidence/result

The applicable CPU tests passed; no CUDA behavior has been measured.

#### Explain it in my own words

`TODO(student): Explain this concept without copying the generated notes.`

#### Questions still open

`TODO(student): List what you still want to understand or test.`

## Flattened causal rows

### Concept

Recovering query position after flattening attention score rows.

#### What I thought before

`TODO(student): Describe your prior mental model in your own words.`

#### What I learned

`TODO(student): Explain what changed in your understanding.`

#### Why it matters

The custom operator sees `[rows, sequence_length]`, not explicit batch, head,
and query axes. It must still exclude every future key correctly.

#### Mental model

Each group of `sequence_length` rows counts query positions from zero again, so
the remainder of `row_index / sequence_length` identifies the query.

#### Where it appears in code

`cuda_attention/reference.py::causal_allowed_mask`, introduced in Commit 007.

#### Experiment demonstrating it

`tests/test_causal_mask.py` checks first, middle, last, and wrapped rows by hand.

#### Evidence/result

CPU tests confirmed exact zeros in masked positions; CUDA remains unimplemented.

#### Explain it in my own words

`TODO(student): Explain this concept without copying the generated notes.`

#### Questions still open

`TODO(student): List what you still want to understand or test.`

## Thread-strided row access

### Concept

Distributing row columns across the threads of one CUDA block.

#### What I thought before

`TODO(student): Describe your prior mental model in your own words.`

#### What I learned

`TODO(student): Explain what changed in your understanding.`

#### Why it matters

A block cannot reduce a row in parallel until each thread owns a disjoint part
of that row. Striding by `blockDim.x` covers arbitrary widths without assuming
that the number of columns equals the block size.

#### Mental model

Thread `t` handles columns `t`, `t + blockDim.x`, `t + 2*blockDim.x`, and so on.
The first pass places neighboring threads on neighboring addresses.

#### Where it appears in code

`csrc/fused_causal_softmax.cu`, introduced in Commit 043.

#### Experiment demonstrating it

CUDA correctness cases for power-of-two and irregular widths are prepared but
have not run without an NVIDIA GPU.

#### Evidence/result

Static inspection confirms the disjoint indexing formula. Runtime correctness
and memory-transaction behavior are not measured.

#### Explain it in my own words

`TODO(student): Explain this concept without copying the generated notes.`

#### Questions still open

`TODO(student): List what you still want to understand or test.`
