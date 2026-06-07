## New Pipeline Project

We want to enhance the current pipeline that creates the implementation of a fine-tuned classifier using weak supervision. The target where we will run the fine-tuning operations is the UCloud platform, with documentation available here: `https://docs.cloud.sdu.dk/`.

The existing pipeline in the Jupyter notebooks should be rewritten from scratch using `.py` scripts that we can run in sequence to perform the operations we want to do. The implementation of the novel pipeline should be done with re-usable functions wrapped in the same place and flexible configuration, following best practices. This is to work beyond the religious classification in the future, also adding other models to train in the pipeline (e.g. pregnancy centers, education, international organizations, etc.)

## Research Material

In `.agents/docs`, several specialized subagents for research saved excellent research handoffs to inform the design of the project. We should always look there for inform the design and use tools like `websearch`, `webfetch`, `context7` and `grep_app_searchGithub` when in need to search for additional examples, technical documentation and actual implementations of the tools in question.

If while consulting documentation and resources it is useful to download locally material, we can download that in a ordered manner in `.agents/archives`.

## Novel Data Input

The new missions and activities for this project are coming out from another sister repository at `~/Documents/Projeects/NonProfitData`. That project is responsible of the panel data pipeline harmonization coming from the data provided by the NCCS at the Urban Institute, and produces the cross-section data in `data/processed/corpus` that are to be used as data input of this project, specifically:

- `missions_cross_section.parquet` containing the `LONGEST_MISSION`, which is the target mission to use for the nonprofits.
- `activities_cross_section.parquet` containing the `CONCATENATED_ACTIVITY`, which is the concatenated activities in the most promising year.

## TODO List notes

1. **Full cleanup of the current status**. We want to store in `src/legacy` the former pipeline, we want to star clean state the new project.

1. **Prompts to annotate religious vs. nonreligious non profits**. We need to work on the prompts based on the research we have done to have a clean identification that relies a clean train/test sub-sample to fine-tune. We want to prepare different versions of prompts as prescribed by the research.

1. **Missions and activities examples**. We want to explore the input data to understand and locate some great examples both of clearly religious and non-religious organizations to use them in the prompts as examples.

1. **Calculate the optimal train/test sample size**. We want to carefully evaluate what is the optimal number of missions to use for the train/test dataset for the best results. We can also cherry-pick the sample for best performance, focusing on clean and long missions.

1. **Create the script for LLM annotation**. We need to decide which model to use if it is not GPT 4 that we were using before, the research can be informing that. We need to prepare the entire script to annotate the selected sample for train/test used for the fine-tuning afterwards.

1. **Create the script for fine-tuning**. We need to fine-tune on different models, based on research of which models are the most desirable for our applications. We need to have state-of-the-art pipeline using most likely the transfomers class from Hugging Face and wwe want full validation metrics to carefully evaluate model performance.

1. **Apply the fine-tuned models to the entire corpus**. We want to use the fine-tuned models to classify the entire corpus and produce visualization of for example cloudwords with the highest n-grams from the embeddings classified as religious or non-religious.
