# Data and responsible use

## CrisisMMD v2.0

CrisisLens is trained and evaluated on [CrisisMMD v2.0](https://crisisnlp.qcri.org/crisismmd.html), released by the
Qatar Computing Research Institute (QCRI): human-labeled tweets and images from seven 2017 disasters.

Its [terms of use](https://crisisnlp.qcri.org/terms-of-use.html) say, in short:

- use the data only for research on humanitarian computing;
- keep the dataset's contents confidential (tweet IDs may be shared);
- delete the data when the research is finished, or when QCRI asks;
- cite the dataset papers in anything derived from it.

## How this project complies

| Rule | What the repo does |
| --- | --- |
| Research use only | Non-commercial portfolio research on disaster response. |
| Keep contents confidential | No tweets or images are committed. `data/` and `outputs/` are gitignored. Notebook outputs are stripped on commit (`.gitattributes` + `nbstripout`). Explanation panels with real posts (`explain.py`) stay in `outputs/`. |
| What is published | Only aggregate results: metrics, class counts, confusion matrices, averaged image reliance, calibration curves. |
| Trained model | Kept in a **private** Hugging Face repo. The terms don't mention model weights, so this is the cautious choice. The public demo loads it with a token and never exposes the files. |
| The demo | Shows **no CrisisMMD content**. The replay is a fictional storm ("Hurricane Delphine") with posts written for this project and public-domain photos from Wikimedia Commons, credited in `demo/scenario/sources.csv`. |
| Deletion | When the project ends: delete `data/crisismmd/` and the private model repo. |
| Citation | Below, and in the README. |

```
Alam, F., Ofli, F., and Imran, M. (2018). CrisisMMD: Multimodal Twitter Datasets from Natural Disasters. ICWSM.
Ofli, F., Alam, F., and Imran, M. (2020). Analysis of Social Media Data using Multimodal Deep Learning for
Disaster Response. ISCRAM.
```

## Known limitations

- **Old data, one language.** English tweets from 2017. Platforms, slang, and photo styles have changed, so accuracy
  on today's posts is unknown and probably lower than the test numbers.
- **Few disasters.** Seven events. The leave-one-disaster-out result (California wildfires) is the best estimate of
  performance on a new disaster.
- **Labels can disagree across modalities.** We use the official agreed-label splits, which drop posts whose text and
  image labels conflict. Real posts are messier.
- **Simulated demo.** The demo posts were written by hand and are cleaner than real tweets, so the demo can look more
  accurate than the model really is. The test-set numbers are the real measure.
- **Location is approximate.** Place names are extracted from text and geocoded; many posts can't be mapped, and
  common names can be mapped to the wrong place.

## Intended use

A triage aid that ranks which posts people look at first. Not for deciding who gets help, not a replacement for
emergency services, and not validated for operational use. Flagged posts should always be reviewed by a person.
