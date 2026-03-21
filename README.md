# ADODAS2026 Baseline

Official baseline implementation for the ADODAS grand challenge (ACMMM 2026).

## Tasks

- **Track A1**: three binary labels for Depression / Anxiety / Stress.
- **Track A2**: 21 ordinal item predictions with scores in `{0, 1, 2, 3}`.

## Environment

Create a conda environment from `envs/adodas.yaml`:

```bash
conda env create -f envs/adodas.yaml
conda activate adodas
```

### Feature directory

The config field `feature_root` should point to the anonymized feature tree. The loader expects:

```text
<feature_root>/<split>/<anon_school>/<anon_class>/<anon_pid>/
|- audio/
|  |- mel_mfcc/<session>/
|  |- vad/<session>/
|  |- ssl_embed/<audio_ssl_model_tag>/<session>/
|  `- egemaps/<session>/
`- video/
     |- headpose_geom/<session>/
     |- face_behavior/<session>/
     |- qc_stats/<session>/
     |- vad_agg/<session>/
     |- body_pose/<session>/
     |- global_motion/<session>/
     `- vision_ssl_embed/<video_ssl_model_tag>/<session>/
```

## Training

```bash
python train.py --task <a1|a2> --config <config_yaml>
```

## Inference

```bash
python infer.py --task <a1|a2> --checkpoint <path_to_best.pt> [--config <config_yaml>] [--split <split_name>] [--output <csv_path>]
```

Typical workflow:

1. Train a model and get `best.pt` under `<output_dir>/runs/<run_name>/checkpoints/`.
2. Run inference using that checkpoint.



## Output Structure

For each run, directories are organized as:

```text
<output_dir>/runs/<run_name>/
|- logs/
|- checkpoints/
|- calibration/
`- submissions/   (created only when submissions are written)
```

## Annotations：
### File Types
- A01：	"The North Wind and the Sun" standardized reading passage.
  - 有一回，北风跟太阳在那儿争论谁的本领大。说着说着，来了一个过路的，身上穿了一件厚袍子。他们俩就商量好了，说谁能先叫这个过路的把他的袍子脱下来，就算是他的本领大。北风就使劲吹起来，拼命地吹。可是，他吹得越厉害，那个人就把他的袍子裹得越紧。到末了儿，北风没辙了，只好就算了。一会儿，太阳出来一晒，那个人马上就把袍子脱了下来。所以，北风不得不承认，还是太阳比他的本领大。
- B01：Please describe how your day went yesterday.
  - 请描述一下，你昨天过的怎么样?
- B02：Please describe your happiest memory from the past week.
  - 请描述一下，现在回想最近一周最开心的记忆?
- B03：Please describe your saddest memory from the past week.
  - 请描述一下，现在回想最近一周最悲伤的记忆?

### Auxiliary Attributes
1. 家庭结构（Family structure）
   1. 1=核心家庭，Nuclear
   2. 2=大家庭, Extended
   3. 3=单亲家庭, Single-parent
   4. 4=重组家庭，Blended
   5. 5=隔代家庭，Skipped-generation
   6. 6=其他，Other
2. 是否是独生子女（Only child status）
   1. Whether the respondent is an only child (1：Yes/0：No).
3. 如非独生子女,是否感受到父母有所偏爱？（Parental favoritism	If not an only child）
   1. 1=偏爱兄弟姐妹，Favoring siblings
   2. 2=无偏爱，No favoritism
   3. 3=偏爱自己，Favoring self
4. 本学期相比上个学期，学习成绩变动情况（Academic performance change Compared with previous semester）: 
   1. 1=进步，Improved, 
   2. 2=退步，Declined, 
   3. 3=稳定，Stable.
5. 本学期相比上个学期，情绪变动情况（Emotional state change Compared with previous semester）: 
   1. 1=变好，Better
   2. 2=变差，Worse
   3. 3=无变化，No change




