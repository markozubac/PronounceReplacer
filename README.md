Triplet extraction with large language models is increasingly used to build knowledge graphs from
text. However, chunk-based extraction frequently produces triplets that contain unresolved pronouns,
leading to ambiguous entities and reduced graph quality. This issue arises when pronoun antecedents
appear outside the processed text segment, preventing the model from grounding the extracted rela-
tions to explicit entities. In this work we investigate how pronoun resolution can improve large
language model–based triplet extraction from chunked documents. We propose several context-aware
strategies that incorporate additional contextual information during extraction in order to replace
pronouns with their corresponding entity mentions. The proposed methods differ in how contex-
tual signals are used, including chunk rewriting, context-aware prompting, and the use of previously
extracted triplets as structured context. Experiments show that the proposed approach substantially
reduces pronoun occurrences in generated triplets. The best-performing strategy decreases pronoun-
based entities by more than 98% compared to the baseline on the HotpotQA dataset, while also
demonstrating consistent improvements on long narrative texts from the NovelQA dataset. We fur-
ther show that cleaner triplets lead to measurable improvements in downstream question answering
when retrieval is performed exclusively over the knowledge graph. These results highlight the impor-
tance of pronoun-aware extraction strategies for building reliable knowledge graphs and improving
graph-based retrieval systems.

Method 1: LLM-Based Pronoun
Resolution via Chunk Rewriting
Method 1 introduces an explicit rewriting step
before re-extraction. If pronouns are detected in
the generated triplets for chunk ci, the system
retrieves up to K = 2 preceding chunks that share
the same question ID (to ensure topical continu-
ity). The LLM is then prompted to rewrite only
the current chunk, replacing pronouns with their
antecedent entity mentions derived from the ear-
lier context. If the antecedent is ambiguous or not
present, the original token is preserved. Triplet
extraction is subsequently applied to the rewrit-
ten chunk using the same strict extraction prompt
as the baseline.
This method separates coreference handling
from relation extraction: the model first pro-
duces a more explicit text representation and then
extracts triplets from it. Algorithm 2 summarizes
the context-aware rewriting strategy employed in
Method 1, where the current chunk is first rewrit-
ten using an LLM to resolve pronoun references
before performing triplet extraction. 


Method 2: Context-Augmented
Regeneration with a Specialized
Extraction Prompt
Method 2 performs re-extraction directly, with-
out rewriting the chunk. Upon pronoun detection,
the system retrieves up to K = 2 preceding
chunks, given they exist and constructs a spe-
cialized prompt in which the LLM is explicitly
instructed to resolve pronouns using the earlier
context and to output triplets with explicit entity
names. The model is asked to extract triplets
from the current chunk only, while using previous
chunks solely for reference resolution.
Compared to Method 1, this approach inte-
grates pronoun grounding into the extraction step,
reducing the number of LLM calls and avoiding
the need for a rewritten intermediate text. Algo-
rithm 3 outlines the regeneration strategy used in
Method 2, in which triplets are re-generated using
a specialized prompt that incorporates preceding
textual context to resolve pronoun ambiguity dur-
ing extraction.

Method 3: Using Prior Triplets
as Structured Context
Method 3 replaces raw textual context with struc-
tured context. After baseline extraction, if pro-
nouns are detected, the system gathers the triplets
extracted from up to K = 2 preceding chunks
(from the same question ID) and supplies them
as contextual evidence in a new prompt. The
LLM is instructed to resolve pronouns in the cur-
rent chunk using only these prior triplets, and to
output explicit entities in the resulting triplets.
This method has two practical motivations:
(i) prior triplets provide compact entity men-
tions that are directly aligned with the tar-
get representation; and (ii) the prompt remains
shorter than when inserting full preceding chunks,
which can be beneficial under context-length con-
straints. Algorithm 4 describes the structured
context strategy of Method 3, where previously
extracted triplets are used as semantic guidance
for resolving pronouns in subsequent extraction
steps. 

Method 4: External Coreference
Model for Rewriting (FastCoref )
In addition to the proposed LLM-based
approaches (Methods 1–3), we evaluate a com-
parative variant that uses an external coreference
resolver for rewriting. When pronouns are
detected, the system concatenates the preceding
context chunks and the current chunk (delimited
by explicit markers), applies the FastCoref model
to obtain coreference clusters, and replaces men-
tions in the current chunk with their antecedents
using rule-based substitution. Triplet extraction
is then performed on the resolved current chunk
using the baseline extraction prompt. This vari-
ant serves as a reference point for evaluating
whether off-the-shelf coreference models can pro-
vide similar improvements without relying on an
LLM to perform the rewriting step.
Across all experiments, identical extraction
constraints and formatting rules were preserved to
ensure that performance differences originate from
contextual strategies rather than prompt struc-
ture. Method 4 uses the same baseline extraction
prompt after external coreference rewriting, and
therefore does not introduce additional variations
in the LLM prompt beyond those listed above.
Algorithm 5 outlines the external coreference-
based rewriting procedure used as a comparative
baseline, where pronoun resolution is performed
using an external model prior to triplet extraction.
