## New Pipeline Project

We want to enhance the current pipeline that creates the implementation of a fine-tuned classifier using weak supervision.

The existing pipeline in the Jupyter notebooks should be rewritten from scratch using `.py` scripts that we can run in sequence to perform the operations we want to do. The implementation of the novel pipeline should be done with re-usable functions wrapped in the same place and flexible configuration, following best practices. This is to work beyond the religious classification in the future, also adding other models to train in the pipeline (e.g. pregnancy centers, education, international organizations, etc.)

## Research Material

In `.agents/docs`, several specialized subagents for research saved excellent research handoffs to inform the design of the project. We should always look there for inform the design and use tools like `websearch`, `webfetch`, `context7` and `grep_app_searchGithub` when in need to search for additional examples, technical documentation and actual implementations of the tools in question.

If while consulting documentation and resources it is useful to download locally material, we can download that in a ordered manner in `.agents/archives`.

## Novel Data Input

The new missions and activities for this project are coming out from another sister repository at `~/Documents/Projeects/NonProfitData`. That project is responsible of the panel data pipeline harmonization coming from the data provided by the NCCS at the Urban Institute, and produces the cross-section data in `data/processed/corpus` that are to be used as data input of this project, specifically:

- `missions_cross_section.parquet` containing the `LONGEST_MISSION`, which is the target mission to use for the nonprofits.
- `activities_cross_section.parquet` containing the `CONCATENATED_ACTIVITY`, which is the concatenated activities in the most promising year.

## Machines

The target where we will run the fine-tuning operations is the UCloud platform, with documentation available here: `https://docs.cloud.sdu.dk/`. We should be able to run on a GPU node consisting of 2x AMD EPYC 9655 CPU@2.6 Ghz, 384 vCPUs, and 2304 GB of DDR5-6400 memory; 8x NVIDIA Blackwell B200-SMX6, 192 GB.

Several apps are available for the execution (https://docs.cloud.sdu.dk/Apps/apps_index.html, but we are able to start a terminal job session with all the requirements and the repository to run the pipeline by configuring it with a starting SSH script.

## TODO List notes

### 1. Upstream Cleanup and Decisions

**Full cleanup of current status**. We want to start fresh with the project, so it would be good to store the former scripts in `src/legacy` and start over.

**Model Choices**. Based on our discoveries, we need to decide on a set of models that are the best choice for our purpose. This implies two separate decisions:

- The model to use for the LLM annotations, which can be a GPT 4 or other better alternatives including open-weighted LLM models. It should not matter for the code implementation and can be done flexibly.
- The model to fine-tune for the classification. We can find a few alternatives, for example one small language model like Llama or better and a set of transformer-based models.

In both cases, we want to let the previous research speak for itself and identify what is the best repertoire of models.

**Ensure Replicability**. Everything stochastic — sampling, splitting, training — takes a fixed seed. We always reason over a single entry-point pipeline script that orchestrates the different phases, but each script can be also run standalone given that the input underneath is available.

### 2. LLM Annotation

**Prompts design**. We need to work on the prompts based on the research done in `.agents/docs` to have a clean identification that delivers a clean train/test sample to use to fine-tune models. We want to consider different version of prompts as prescribed by the research, test them on 30-50 hand-selected examples spanning the distribution before committing to a full labeling run.

**Missions and activities examples**. For the selection of the examples, we should carefully evaluate the input data to understand great missions/activities to use that are clearly religious and non-religious based on the prompts that we engineered.

**Sample selection**. We need to carefully find the optimal number of missions to use for the train/test dataset for the fine-tuning. We can cherry-pick the sample for best performance, stratify for example by NTEEs major groups so that rare categories are represented, where we can get the NTEEs from the unified BMF. Here as well, for the sampling we should use a fixed seed and be able to identify back the nonprofits to EIN codes.

**Output storage**. We want to store the full LLM response (reasoning + final label), not just the extracted label. You will want the reasoning later when auditing errors or explaining the classification decision to reviewers.

**Quality check**. Manually review some labeled examples. Compute agreement between the human judgement and the LLM labels. If agreement is below ~85%, revise the prompt and re-label. This is the only human-in-the-loop step, so it matters.

Once quality is acceptable, version the labeled file and do not re-label. Re-running labeling with a newer model version changes your data without a clear audit trail.

### 3. Training Setup

We need to create the full pipeline for training no matter the models picked. The research will guide us on the best solutions also with respect to the actual data, optimizing class imbalance, training and reproducibility. We want a state-of-the-art pipeline.

The sample needs to be 3-way stratified adequately between train, validation and test. The test set is held out until we have the final models.

We probably want to use the standard HuggingFace Trainer where possible, saving checkpoints on best validation F1, and logging all train loss, val loss, and val F1 at every epoch to a file, not just to the terminal.

### 4. Evaluation

We want to evaluate each model following research recommendation, setting acceptance criteria before unlocking the test set.

### 5. Inference at scale

We want to run the final models over all mission records in batches, with GPU batching with some packages like DataLoader or datasets.map if supported by the UCloud architecture. Always store confidence scores, write predicted label and predictive probability for the positive class for every record. Do not store only the hard label. Always keep EIN2, which are the keys to merge into the upstream data repository. Record the model version, checkpoint hash and inference date in the output file as metadata columns. This makes the results reproducible even if the model is retrained later.

### 6. Visualization

From the results we want to produce visualization with cloudwords with the highest n-grams from the embeddings classified as religious and non-religious, and all of other sorts of visualizations of evaluation metrics and classification results.
