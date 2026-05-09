"""Generate reference WAVs locally via Qwen3-TTS **VoiceDesign** model.

This is the **only** prompt-creation flow in the project. Runs Qwen3-TTS's
VoiceDesign variant locally and synthesises a 5-15 s WAV per voice from
a natural-language description — fully on-device, no external account
or per-character billing. Output files plug into the runtime path
unchanged — the live TTS backend stays on Qwen3-TTS Base + zero-shot
cloning of these WAVs.

Why VoiceDesign for *prompt generation* but Base for *playback*:
- VoiceDesign lets us iterate on voices without touching cloud APIs
  or paying per character.
- Base + ref_audio is deterministic (same WAV → same voice every
  request); VoiceDesign's outputs vary per sample. Generating once and
  freezing the WAV gives the runtime stable, repeatable voices.

Workflow:
    # 1. Download a VoiceDesign variant of Qwen3-TTS (one-off, ~1.7GB).
    #    Note: as of 2026-04, VoiceDesign is only published for the
    #    1.7B model — there's no 0.6B-VoiceDesign and no 4bit quantised
    #    variant on HF. Use the 8bit MLX build:
    HF_ENDPOINT=https://hf-mirror.com python -c "
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id='mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit',
        local_dir='pretrained_models/Qwen3-TTS-VoiceDesign',
        max_workers=4,
    )"

    # 2. Generate the prompt library (45 voices, several minutes on M4):
    python -m scripts.generate_voicedesign_voices \\
        --model-dir pretrained_models/Qwen3-TTS-VoiceDesign

    # 3. (Optional) Re-roll a single voice you don't like:
    python -m scripts.generate_voicedesign_voices \\
        --model-dir pretrained_models/Qwen3-TTS-VoiceDesign \\
        --ids ancient_male_adult,fierce_female_elder --regenerate

The script also updates ``data/voices/speakers.json`` to register the
new ``zs:vd_<tag>`` entries (uses the same TAG_TO_ATTRS mapping as the
Volcengine path so personality/gender/age stays consistent).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from mlx.nn import Module

from app.config import get_settings
from app.core.enums import Age, Gender, Personality
from app.core.models import Speaker
from app.services.tts_qwen3 import _collect_audio_wav


log = logging.getLogger("vd_voices")


# ---------------------------------------------------------------------------
# Tag → attribute mapping. Owned here because this is the only consumer left
# (the prior Volcengine-catalog flow has been removed). 64 entries cover the
# full prompt library — VoiceDesign tags that lack a row here surface as
# warnings and don't get a `Speaker` written into ``speakers.json``.
# ---------------------------------------------------------------------------
TAG_TO_ATTRS: dict[str, tuple[Gender, Age, list[Personality]]] = {
    # ---- 旁白 / 解说 ----
    "narrator_male_mature":     (Gender.MALE,    Age.ADULT,  [Personality.CALM, Personality.MATURE]),
    "suspense_male_adult":      (Gender.MALE,    Age.ADULT,  [Personality.SERIOUS, Personality.MELANCHOLY]),
    "magnetic_male_adult":      (Gender.MALE,    Age.ADULT,  [Personality.MATURE, Personality.GENTLE]),
    "calm_male_adult":          (Gender.MALE,    Age.ADULT,  [Personality.CALM]),
    "wise_male_elder":          (Gender.MALE,    Age.ELDER,  [Personality.WISE, Personality.CALM]),
    "cold_male_adult":          (Gender.MALE,    Age.ADULT,  [Personality.COLD, Personality.MATURE]),
    "narrator_female_adult":    (Gender.FEMALE,  Age.ADULT,  [Personality.GENTLE, Personality.CALM]),
    "wise_female_adult":        (Gender.FEMALE,  Age.ADULT,  [Personality.WISE, Personality.GENTLE]),

    # ---- 男·少年/青年 ----
    "youth_male_teen":          (Gender.MALE,    Age.TEEN,   [Personality.DETERMINED, Personality.BRAVE]),
    "boyish_male_teen":         (Gender.MALE,    Age.TEEN,   [Personality.NAIVE, Personality.KIND]),
    "cheerful_male_youth":      (Gender.MALE,    Age.YOUTH,  [Personality.CHEERFUL, Personality.PASSIONATE]),
    "energetic_male_youth":     (Gender.MALE,    Age.YOUTH,  [Personality.CHEERFUL, Personality.BRAVE]),
    "gentle_male_youth":        (Gender.MALE,    Age.YOUTH,  [Personality.GENTLE, Personality.KIND]),
    "warm_male_youth":          (Gender.MALE,    Age.YOUTH,  [Personality.GENTLE, Personality.MATURE]),
    "elegant_male_youth":       (Gender.MALE,    Age.YOUTH,  [Personality.CALM, Personality.MATURE]),
    "fresh_male_youth":         (Gender.MALE,    Age.YOUTH,  [Personality.CHEERFUL, Personality.NAIVE]),

    # ---- 男·中年 ----
    "arrogant_male_adult":      (Gender.MALE,    Age.ADULT,  [Personality.ARROGANT, Personality.COLD]),
    "dub_male_adult":           (Gender.MALE,    Age.ADULT,  [Personality.SERIOUS, Personality.MATURE]),
    "podcast_male_adult":       (Gender.MALE,    Age.ADULT,  [Personality.CALM, Personality.GENTLE]),
    "lazy_male_youth":          (Gender.MALE,    Age.YOUTH,  [Personality.PLAYFUL, Personality.CUNNING]),

    # ---- 女·少女/青年 ----
    "girl_female_teen":         (Gender.FEMALE,  Age.TEEN,   [Personality.GENTLE, Personality.KIND]),
    "cute_female_teen":         (Gender.FEMALE,  Age.TEEN,   [Personality.NAIVE, Personality.PLAYFUL]),
    "sweet_female_youth":       (Gender.FEMALE,  Age.YOUTH,  [Personality.GENTLE, Personality.CHEERFUL]),
    "sweet_female_teen":        (Gender.FEMALE,  Age.TEEN,   [Personality.GENTLE, Personality.NAIVE]),
    "fresh_female_youth":       (Gender.FEMALE,  Age.YOUTH,  [Personality.CHEERFUL, Personality.NAIVE]),
    "playful_female_youth":     (Gender.FEMALE,  Age.YOUTH,  [Personality.PLAYFUL, Personality.CHEERFUL]),
    "energetic_female_youth":   (Gender.FEMALE,  Age.YOUTH,  [Personality.CHEERFUL, Personality.PASSIONATE]),
    "cheerful_female_adult":    (Gender.FEMALE,  Age.ADULT,  [Personality.CHEERFUL, Personality.KIND]),
    "gentle_female_youth":      (Gender.FEMALE,  Age.YOUTH,  [Personality.GENTLE, Personality.CALM]),
    "wise_female_youth":        (Gender.FEMALE,  Age.YOUTH,  [Personality.WISE, Personality.GENTLE]),

    # ---- 女·成熟 ----
    "cold_female_adult":        (Gender.FEMALE,  Age.ADULT,  [Personality.COLD, Personality.ARROGANT]),
    "tender_female_adult":      (Gender.FEMALE,  Age.ADULT,  [Personality.GENTLE, Personality.PASSIONATE]),
    "charming_female_adult":    (Gender.FEMALE,  Age.ADULT,  [Personality.PASSIONATE, Personality.MATURE]),
    "graceful_female_adult":    (Gender.FEMALE,  Age.ADULT,  [Personality.GENTLE, Personality.MATURE]),
    "mature_female_adult":      (Gender.FEMALE,  Age.ADULT,  [Personality.MATURE, Personality.CALM]),
    "kind_female_adult":        (Gender.FEMALE,  Age.ADULT,  [Personality.KIND, Personality.GENTLE]),

    # ---- 角色扮演 / 古风 ----
    "noble_male_adult":         (Gender.MALE,    Age.ADULT,  [Personality.CALM, Personality.SERIOUS]),
    "ancient_male_adult":       (Gender.MALE,    Age.ADULT,  [Personality.SERIOUS, Personality.COLD]),
    "philosopher_male_adult":   (Gender.MALE,    Age.ADULT,  [Personality.WISE, Personality.CALM]),
    "ancient_female_youth":     (Gender.FEMALE,  Age.YOUTH,  [Personality.GENTLE, Personality.MATURE]),
    "fierce_female_elder":      (Gender.FEMALE,  Age.ELDER,  [Personality.FIERCE, Personality.ARROGANT]),

    # ---- 童声 / 故事（童声为中性，按性格区分而非性别）----
    "child_lively":             (Gender.NEUTRAL, Age.CHILD,  [Personality.PLAYFUL, Personality.BRAVE]),
    "child_gentle":             (Gender.NEUTRAL, Age.CHILD,  [Personality.GENTLE, Personality.NAIVE]),
    "child_timid":              (Gender.NEUTRAL, Age.CHILD,  [Personality.TIMID, Personality.NAIVE]),
    "child_cheerful":           (Gender.NEUTRAL, Age.CHILD,  [Personality.CHEERFUL, Personality.KIND]),
    "child_clever":             (Gender.NEUTRAL, Age.CHILD,  [Personality.CUNNING, Personality.PLAYFUL]),
    "child_melancholy":         (Gender.NEUTRAL, Age.CHILD,  [Personality.MELANCHOLY, Personality.GENTLE]),
    "storybook_female":         (Gender.FEMALE,  Age.ADULT,  [Personality.GENTLE, Personality.PLAYFUL]),
    "kidstory_female":          (Gender.FEMALE,  Age.ADULT,  [Personality.GENTLE, Personality.CHEERFUL]),

    # ---- 怯懦 / 忧郁（覆盖 timid / melancholy 性格 tag）----
    "timid_male_youth":         (Gender.MALE,    Age.YOUTH,  [Personality.TIMID, Personality.GENTLE]),
    "timid_female_youth":       (Gender.FEMALE,  Age.YOUTH,  [Personality.TIMID, Personality.NAIVE]),
    "melancholy_male_youth":    (Gender.MALE,    Age.YOUTH,  [Personality.MELANCHOLY, Personality.CALM]),
    "melancholy_female_youth":  (Gender.FEMALE,  Age.YOUTH,  [Personality.MELANCHOLY, Personality.GENTLE]),

    # ---- 补 cell 内空缺（成年男 暖/热血缺位、女青年 冷酷缺位、成年女 心机/果敢缺位）----
    "passionate_male_adult":    (Gender.MALE,    Age.ADULT,  [Personality.PASSIONATE, Personality.BRAVE]),
    "kind_male_adult":          (Gender.MALE,    Age.ADULT,  [Personality.KIND, Personality.GENTLE]),
    "cold_female_youth":        (Gender.FEMALE,  Age.YOUTH,  [Personality.COLD, Personality.ARROGANT]),
    "cunning_female_adult":     (Gender.FEMALE,  Age.ADULT,  [Personality.CUNNING, Personality.MATURE]),
    "brave_female_adult":       (Gender.FEMALE,  Age.ADULT,  [Personality.BRAVE, Personality.DETERMINED]),

    # ---- 老年扩充：从 2 条扩到 8 条（4 男 + 4 女）----
    "kind_male_elder":          (Gender.MALE,    Age.ELDER,  [Personality.KIND, Personality.GENTLE]),
    "fierce_male_elder":        (Gender.MALE,    Age.ELDER,  [Personality.FIERCE, Personality.COLD]),
    "cunning_male_elder":       (Gender.MALE,    Age.ELDER,  [Personality.CUNNING, Personality.MATURE]),
    "wise_female_elder":        (Gender.FEMALE,  Age.ELDER,  [Personality.WISE, Personality.CALM]),
    "kind_female_elder":        (Gender.FEMALE,  Age.ELDER,  [Personality.KIND, Personality.GENTLE]),
    "melancholy_female_elder":  (Gender.FEMALE,  Age.ELDER,  [Personality.MELANCHOLY, Personality.GENTLE]),
}


def merge_library(
    existing_path: Path,
    new_speakers: list[Speaker],
    *,
    clean: bool = False,
) -> list[Speaker]:
    """Merge new speakers into existing speakers.json.

    With ``clean=True``, also drop entries whose speaker_id has no
    ``zs:`` / ``sft:`` prefix — those crash the runtime backend's
    speaker-id parser, so they're never useful.
    """
    table: dict[str, Speaker] = {}
    if existing_path.exists():
        raw = json.loads(existing_path.read_text(encoding="utf-8"))
        for item in raw:
            spk = Speaker.model_validate(item)
            if clean and not (
                spk.speaker_id.startswith("zs:")
                or spk.speaker_id.startswith("sft:")
            ):
                log.info("clean: removing %s (no zs:/sft: prefix)", spk.speaker_id)
                continue
            table[spk.speaker_id] = spk
    for spk in new_speakers:
        table[spk.speaker_id] = spk
    return sorted(table.values(), key=lambda s: s.speaker_id)


def write_library(path: Path, speakers: list[Speaker]) -> None:
    data = [s.model_dump(mode="json") for s in speakers]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


# Reference audio's CONTENT TYPE biases the clone's delivery style.
# Four samples: pure narration for narrators (flat/storytelling), then
# age-appropriate dramatic dialogue for character voices. Putting
# "我等了十年" in a 7-year-old's reference would produce an absurd
# child reading an adult monologue — wrong delivery cues all round.
# Per-tag reference text for VoiceDesign generation.
#
# **Why per-tag unique short ref_texts (rewritten 2026-04-28)**:
#
# The previous scheme grouped 51 of 64 tags under a single dramatic
# adult sample text ("「我等了十年...只有强者，和弱者。」"). At
# inference time, **all 51 voices bled the tail of that ref_text into
# every output** — verified by listening to a 38-variant punctuation
# benchmark where every single sample produced "...只有强者和弱者，
# 他望着窗外..." regardless of target text.
#
# Mechanism: zero-shot voice clone concatenates `[ref_text_tokens] +
# [target_text_tokens]` with no separator. When ref_text is long /
# dramatic / shared across speakers, it forms a strong attention
# attractor that the model can't cross at the boundary. The "current
# text pointer" stays anchored in the ref portion at generation start,
# so the first audio frames produced are continuations of ref_text.
#
# A 5-speaker test with short, neutral, per-speaker-unique ref_texts
# produced 0/10 bleed. So this table replaces the shared samples with
# 64 unique declarative sentences — short (12-18 chars + period),
# neutral mood (no quotes / exclamations / dramatic phrasing), each
# distinct enough to not reinforce a single attention pattern across
# speakers.
#
# Age tier is preserved softly: child/teen tags get age-appropriate
# scenes (school, toys, play) since the acoustic delivery still
# benefits from age-matched prosody, but no longer with dramatic
# emotional outbursts.
REF_TEXT_FOR_TAG: dict[str, str] = {
    # narrator (2)
    "narrator_male_mature":   "故事要从一个寻常的清晨说起。",
    "narrator_female_adult":  "那是个阳光很好的下午，风也不大。",

    # adult / elder / youth — neutral declarative (49)
    "ancient_female_youth":   "山间的清风吹过竹林，树叶轻轻摆着。",
    "ancient_male_adult":     "雨水沿着屋檐缓缓落下，打在青石板上。",
    "arrogant_male_adult":    "桌前摆着一盏旧灯，灯光柔和地照在书页上。",
    "brave_female_adult":     "远处的河水在晨雾中静静流淌。",
    "calm_male_adult":        "庭院里的梅花已经开了几枝。",
    "charming_female_adult":  "屋外飘起细雪，覆盖了石阶。",
    "cheerful_female_adult":  "他放下手中的笔，轻轻揉了揉额头。",
    "cheerful_male_youth":    "街角的茶馆里坐着不少人。",
    "cold_female_adult":      "钟声从远处传来，回荡在山谷之间。",
    "cold_female_youth":      "她端起杯子，看着窗外发了一会儿呆。",
    "cold_male_adult":        "走廊尽头的灯还亮着。",
    "cunning_female_adult":   "风吹动了案几上的纸页。",
    "cunning_male_elder":     "屋外的雪还在静静地下着。",
    "dub_male_adult":         "院墙外的桂花香气阵阵袭来。",
    "elegant_male_youth":     "他望着远方的山影，没有说话。",
    "energetic_female_youth": "月光透过纱窗，落在书桌一角。",
    "energetic_male_youth":   "江面上飘着几只白鹭。",
    "fierce_female_elder":    "窗台上的兰花刚刚开放。",
    "fierce_male_elder":      "屋檐下挂着一串风铃。",
    "fresh_female_youth":     "巷子深处有人在慢慢走过。",
    "fresh_male_youth":       "长廊上的灯笼随风轻轻摇晃。",
    "gentle_female_youth":    "池塘里的荷叶在风中轻轻摇着。",
    "gentle_male_youth":      "阶前的青苔已经厚厚一层。",
    "graceful_female_adult":  "他在桌边坐下，开始整理那一摞旧信。",
    "kind_female_adult":      "门外有脚步声，由远而近。",
    "kind_female_elder":      "茶汤在杯中静静地冷着。",
    "kind_male_adult":        "阳光穿过云层洒在湖面上。",
    "kind_male_elder":        "老人慢慢走过石桥，停了下来。",
    "lazy_male_youth":        "厨房里炖着一锅热汤。",
    "magnetic_male_adult":    "树梢上的鸟雀已经飞远了。",
    "mature_female_adult":    "她翻开桌上的诗集，默默读了起来。",
    "melancholy_female_elder": "屋里只剩下钟摆在缓缓地摆着。",
    "melancholy_female_youth": "廊下的雨声整夜不停。",
    "melancholy_male_youth":  "河边的柳条被风轻轻拂动。",
    "noble_male_adult":       "她拿起绣花针，继续未完的图样。",
    "passionate_male_adult":  "烛火在风中摇晃了几下。",
    "philosopher_male_adult": "远山的轮廓在暮色里渐渐模糊。",
    "playful_female_youth":   "他抬头看了看墙上的旧画。",
    "podcast_male_adult":     "杯中的茶叶慢慢沉到底部。",
    "suspense_male_adult":    "庭外有几个孩童在玩着。",
    "sweet_female_youth":     "她在花瓶里插了几枝新摘的腊梅。",
    "tender_female_adult":    "她端着一碗粥走了进来。",
    "timid_female_youth":     "她把窗帘轻轻拉开了一些。",
    "timid_male_youth":       "他翻开旧书，纸页已经发黄。",
    "warm_male_youth":        "他在窗前点上了第二支烛火。",
    "wise_female_adult":      "屋外的雨已经下了整整一天。",
    "wise_female_elder":      "她把信放进木匣，轻轻合上。",
    "wise_female_youth":      "她沿着河岸走了一段路。",
    "wise_male_elder":        "他在门前的老槐树下坐了下来。",

    # teen (5)
    "boyish_male_teen":       "操场上的同学们正在玩着。",
    "cute_female_teen":       "她坐在教室最后一排，看着窗外。",
    "girl_female_teen":       "桌上的笔记本翻到了新的一页。",
    "sweet_female_teen":      "她在书架前停了停，又走开了。",
    "youth_male_teen":        "他背着书包，朝学校的方向走去。",

    # child / story (8)
    "child_cheerful":         "小孩子坐在屋檐下数着雨点。",
    "child_clever":           "他踮起脚去够架子上的玩具。",
    "child_gentle":           "她把彩纸折成纸船，放进水里。",
    "child_lively":           "他抱着一只小猫，在院子里走着。",
    "child_melancholy":       "他蹲在地上看蚂蚁搬家。",
    "child_timid":            "她数着手指，认真地比画着。",
    "kidstory_female":        "院里的小孩在沙地上玩着。",
    "storybook_female":       "他轻声哼着不知名的小调。",
}

# Defensive: catch typos / missing tags at script start instead of
# silently using a fallback.
_FALLBACK_REF_TEXT = "屋外的天色渐渐暗了下来。"


def sample_text_for(tag: str) -> str:
    text = REF_TEXT_FOR_TAG.get(tag)
    if text is None:
        log.warning(
            "no REF_TEXT_FOR_TAG entry for tag=%r; using fallback. "
            "Add a unique sentence to the table to avoid bleed.",
            tag,
        )
        return _FALLBACK_REF_TEXT
    return text


# Each entry is (tag, label, description). The tag must exist in
# ``TAG_TO_ATTRS`` above — that's where we get gender/age/personality.
# The description is the natural-language instruction handed to
# ``model.generate_voice_design``. Keep it specific (gender, age, pitch,
# timbre, scenario) so VoiceDesign has enough signal to land on a
# distinct voice.
VD_CATALOG: list[dict[str, str]] = [
    # ---- 旁白 / 解说 ----
    {"tag": "narrator_male_mature",  "label": "中年男旁白",
     "description": "三十五岁左右的成年男声，中音区清亮干净，音色不低沉也不夸张。讲故事的口吻，平实自然，像朋友在你面前娓娓道来，而不是在舞台上朗诵诗歌。语调保持平稳，情绪克制少起伏，吐字清晰但不刻意强调，语速从容均匀，适合长篇小说类有声书的平静讲述。"},
    {"tag": "suspense_male_adult",   "label": "悬疑解说男",
     "description": "中年男声，声线低沉冷峻略带气声，语调缓慢压抑、句间停顿带悬念感。悬疑剧反派或警匪片审讯室那种戏剧化腔调——重音落在关键悬念词上，每个字都像在埋暗示。要演绎出悬疑张力，不是日常陈述的平淡。"},
    {"tag": "magnetic_male_adult",   "label": "磁性中年男",
     "description": "中年男声，胸腔共鸣强、声音温润磁性。深夜情感电台主持人那种戏剧化的磁性腔——气息悠然，重音常落在情感词上，句尾偶有低音拖音，营造亲密感。要演绎出富有魅力的成熟感，不是干巴巴的播报。"},
    {"tag": "calm_male_adult",       "label": "沉稳中年男",
     "description": "中年男声，音色平实沉稳，但带戏剧张力——重音清晰、关键句子有节奏感，像古装剧里运筹帷幄的谋士说话那样。看似平静却字字千斤，不是日常无趣的平铺直叙。"},
    {"tag": "wise_male_elder",       "label": "智慧长者男",
     "description": "五十多岁的老年男声，嗓音温厚带微微沙哑。武侠或修仙小说里世外高人的腔调——语速缓慢但字字精到，重音落在哲思的关键词上，偶有意味深长的停顿和叹息。要演绎出经历世事的厚重感，不是普通老年人闲话家常。"},
    {"tag": "cold_male_adult",       "label": "高冷男声",
     "description": "中年男声，声线冷峻锋利、吐字干脆。古偶剧里禁欲系男主或职场剧里冷面 boss 的腔调——语调平直却带威压，句尾下沉，重音故意短促压抑。要演绎出戏剧化的距离感和不容置疑感，不是日常社交的平淡。"},
    {"tag": "narrator_female_adult", "label": "流畅女旁白",
     "description": "三十岁左右的成年女声，中音偏暖，音色清润干净，不娇柔也不刻意稳重。讲故事的口吻，自然亲切，像姐姐在你耳边慢慢讲述，而不是在课堂上朗读课文。语调平稳少起伏，情绪克制，吐字清晰但不刻意强调，语速从容均匀，适合长篇小说类有声书的安静讲述。"},
    {"tag": "wise_female_adult",     "label": "知性女声",
     "description": "三十多岁的成年女声，中音偏低、嗓音温润有质感。古装剧才女或现代剧女教授的腔调——语速沉稳、吐字考究、重音落在思考的关键词上，整体有书卷气和思辨感。要演绎出戏剧化的睿智知性感，不是新闻播报的平铺。"},

    # ---- 男·少年/青年 ----
    {"tag": "youth_male_teen",       "label": "少年男声",
     "description": "十六七岁的少年男声，音色清亮带青涩感，语调中偏快、起伏明显。热血动漫里少年主角的 declamatory 表演腔——重音落在表决心的词上，关键句子带向上挑的语气，像在喊『我一定要变强』那种戏剧化感。要演绎出少年英雄气，不是日常同学间的随意聊天。"},
    {"tag": "boyish_male_teen",      "label": "邻家男孩",
     "description": "十六七岁的少年男声，已过变声期，音色比成年男声明亮但具备男声厚度和胸腔共鸣。校园剧里好朋友配角的味道——说话明快带笑意、语调上扬开朗，像青春剧主角身边那个总能逗笑大家的男同学。要演绎出朋友间鲜活的戏剧感，不是日常无聊的对话。"},
    {"tag": "cheerful_male_youth",   "label": "阳光青年男",
     "description": "二十岁出头的青年男声，声音明亮温暖，语调上扬。少年漫画阳光主角那种 declamatory 的开朗——重音常落在情绪饱满的词上、句尾上挑，整体带感染力和号召力。要演绎出令人想跟随的明亮感，不是普通同学的随意聊天。"},
    {"tag": "energetic_male_youth",  "label": "活力小哥",
     "description": "二十岁出头的青年男声，音色清脆有力，语速明快、节奏感强。体育解说或综艺主持那种 high-energy 表演腔——重音夸张外放、语调起伏大，能把听众情绪带起来。要演绎出燃的感觉，不是日常聊天的平淡。"},
    {"tag": "gentle_male_youth",     "label": "温柔青年男",
     "description": "二十多岁的青年男声，声线温柔细腻略带气声，语调缓慢轻柔，吐字软。言情剧男主告白时的修饰过的温柔——句尾常带绵长拖音，像在哄人。要演绎出戏剧化的温柔感，不是日常没心没肺的随意。"},
    {"tag": "warm_male_youth",       "label": "温暖男声",
     "description": "二十六七岁的青年男声，中音偏暖、嗓音宽厚有质感。校园剧里温柔学长的配音——比同龄人多一份成熟稳重，语速从容，重音落在表达关心的词上。要演绎出戏剧化的温暖感，让听者想要依靠的那种声音，不是同事间的客套寒暄。"},
    {"tag": "elegant_male_youth",    "label": "儒雅青年男",
     "description": "二十多岁的青年男声，音色清润儒雅，吐字工整。古装剧里世家公子的腔调——语调内敛但每个字都咬得讲究、带韵味，像在念诗。要演绎出戏剧化的书卷气和古典感，不是现代日常那种随便。"},
    {"tag": "fresh_male_youth",      "label": "清爽男大学生",
     "description": "二十岁左右的男大学生声线，干净清爽，没有沧桑感。青春偶像剧男主角的声音——语速轻快活泼、语调上扬带笑意，整体明朗有戏剧化的青春感。要演绎出让人心动的初恋感，不是普通学生的平淡聊天。"},

    # ---- 男·中年 ----
    {"tag": "arrogant_male_adult",   "label": "傲娇霸总",
     "description": "三十岁左右的成年男声，低沉富磁性。影视剧霸道总裁那种戏剧化的傲慢腔——句尾习惯往下压，重音落在自我标榜的词上，整体带 PUA 式压迫感和居高临下的玩味。要演绎出戏剧化的傲慢张力，不是日常职场对话的平淡。"},
    {"tag": "dub_male_adult",        "label": "译制片男配",
     "description": "中年男声，明显的老式译制片腔调——语速沉稳、字字工整带朗诵感，元音拉长、轻声字也咬清楚。像八十年代上译厂男配音演员的风格，戏剧张力十足。要演绎出标志性的『译制腔』，不是现代日常说话。"},
    {"tag": "podcast_male_adult",    "label": "深夜播客男",
     "description": "中年男声，嗓音温暖低沉，气息悠然带感染力。深夜情感电台的男主播——语速放慢、句间留白、重音常落在情感词上，营造亲密的对话感。要演绎出戏剧化的温暖陪伴感，不是新闻播报的平铺。"},
    {"tag": "lazy_male_youth",       "label": "懒散青年男",
     "description": "二十多岁的青年男声，语调慵懒散漫、语速偏慢、句尾常拖音。喜剧或日剧里废柴男配的腔调——带戏剧化的不在乎和玩世不恭，重音故意落在不该重的位置制造慵懒感。要演绎出戏剧化的痞气，不是真实的散漫。"},

    # ---- 女·少女 / 青年 ----
    {"tag": "girl_female_teen",      "label": "邻家女孩",
     "description": "十五六岁少女声线，音色清亮带青涩感。校园剧女主角的好朋友配角——语调上扬带笑意、说话明快有节奏，重音落在情感词上，整体鲜活灵动。要演绎出戏剧化的少女感，不是日常聊天的平淡。"},
    {"tag": "cute_female_teen",      "label": "萌妹",
     "description": "十几岁的少女声线，音调偏高、起伏夸张。日漫或国漫里萌系女角色的腔调——撒娇感明显、句尾常上扬带卖萌拖音，重音故意夸张。要演绎出戏剧化的萌系卡通感，不是真实少女的日常说话。"},
    {"tag": "sweet_female_youth",    "label": "甜美女青年",
     "description": "二十出头的青年女声，音色甜美悦耳带笑意。言情剧女主角告白时的腔调——语调温柔上扬、句尾带绵长拖音，吐字软糯，重音落在情感词上让人心化。要演绎出戏剧化的甜美感，不是同事间的客套。"},
    {"tag": "sweet_female_teen",     "label": "甜美少女",
     "description": "十六七岁少女声线，嗓音清甜柔软。校园偶像剧学妹角色——语调起伏自然带青涩感、笑意盈盈，关键词带少女特有的羞涩拖音。要演绎出戏剧化的少女初恋感，不是真实日常的随意。"},
    {"tag": "fresh_female_youth",    "label": "清新女青年",
     "description": "二十多岁青年女声，音色清新干净带稚嫩感。文艺片或校园清新风女主角的腔调——语速明快、语调上扬、吐字干净不修饰，像清晨的风。要演绎出戏剧化的青春清新感，不是死气沉沉的日常陈述。"},
    {"tag": "playful_female_youth",  "label": "俏皮女声",
     "description": "二十多岁青年女声，语气俏皮灵动、语调起伏夸张。综艺或脱口秀女主持的腔调——重音故意落在出人意料的位置制造逗趣感、句尾常带挑逗的上扬，整体俏皮顽皮。要演绎出戏剧化的逗趣感，不是普通日常聊天。"},
    {"tag": "energetic_female_youth", "label": "爽快女青年",
     "description": "二十多岁青年女声，嗓音明亮有力、语速明快。运动番女主或女团 C 位那种 high-energy 表演腔——重音夸张外放、语调起伏大、富有号召力。要演绎出戏剧化的燃感和带动力，不是日常对话的平淡。"},
    {"tag": "cheerful_female_adult", "label": "开朗姐姐",
     "description": "二十八九岁的成年女声，音色明亮温暖、笑意盈盈。都市轻喜剧里开朗女主或女团 mentor 姐姐的腔调——语调上扬、语速偏快、情绪外放、重音常落在感叹词上，给人天生快乐源泉的感觉。要演绎出戏剧化的明朗感，不是日常应酬的客套。"},
    {"tag": "gentle_female_youth",   "label": "温柔女青年",
     "description": "二十多岁青年女声，声线轻柔细腻略带气声。治愈系动漫女主角或言情剧温柔女配的腔调——语速慢、句尾带绵长拖音、吐字软糯如棉，重音落在表关心的词上。要演绎出戏剧化的治愈温柔感，不是日常说话的客套。"},
    {"tag": "wise_female_youth",     "label": "智慧女青年",
     "description": "二十多岁青年女声，音色温和稳重带书卷气。古装剧才女或现代剧资深心理咨询师的腔调——语调内敛但每个字都咬得讲究、重音落在思考的关键词上、节奏带停顿感。要演绎出戏剧化的睿智感，不是普通日常的随意。"},

    # ---- 女·成熟 / 御姐 ----
    {"tag": "cold_female_adult",     "label": "高冷御姐",
     "description": "三十岁左右的成熟女声，低沉冷艳带磁性。宫斗剧手握权柄的反派妃嫔或商战剧女总裁的腔调——语调下沉缓慢、每个字咬清楚带冷硬感、重音故意压低制造威压，气息克制不带笑意。要演绎出戏剧化的杀气和距离感，不是日常职场的平淡。"},
    {"tag": "tender_female_adult",   "label": "柔美女友",
     "description": "三十出头的成年女声，嗓音柔美绵软略带气声。言情剧女主与男友独处时的温柔腔——语调缓慢、句尾上扬带绵长拖音、吐字软糯，重音落在亲密称呼或感情词上。要演绎出戏剧化的恋人甜蜜感，不是同事间的客套寒暄。"},
    {"tag": "charming_female_adult", "label": "魅惑女声",
     "description": "三十岁左右的成熟女声，音色低沉性感、气声明显。年代剧上海滩名媛或谍战剧女间谍的腔调——语速放慢有诱惑感、句尾常带气声拖音、重音故意落在情绪词上勾人心弦。要演绎出戏剧化的魅惑感，不是日常说话的随意。"},
    {"tag": "graceful_female_adult", "label": "温柔淑女",
     "description": "三十岁左右的成熟女声，音色温润优雅。古装剧大家闺秀或后宫剧端庄皇后的腔调——语速从容、吐字考究有古典韵味、重音落在礼仪相关的词上，整体仪态万方。要演绎出戏剧化的端庄贵气，不是现代日常的随便。"},
    {"tag": "mature_female_adult",   "label": "成熟女声",
     "description": "三十多岁的成熟女声，音色稳重大气。职场剧女强人或商战剧女 CEO 的腔调——语调平稳有威严、吐字干脆利落、重音落在决策相关词上，整体不容置疑。要演绎出戏剧化的领导气场，不是普通办公室对话的平实。"},
    {"tag": "kind_female_adult",     "label": "温柔妈妈",
     "description": "三十多岁的成熟女声，声音温暖宽厚带母性慈爱感。家庭剧温柔母亲或校园剧温柔老师的腔调——语速舒缓、吐字清晰有耐心、重音落在表关爱的词上、句尾偶有放轻的拖音。要演绎出戏剧化的母爱感，不是同事间的客套。"},

    # ---- 角色扮演 / 古风 ----
    {"tag": "noble_male_adult",      "label": "贵族男声",
     "description": "三十多岁的成年男声，音色清贵优雅带英气。古装剧或仙侠剧里世家公子/仙门弟子的腔调——语调从容略带高傲，吐字考究有礼制感，每个字都咬得讲究。要演绎出戏剧化的贵气和疏离感，不是现代日常的随便。"},
    {"tag": "ancient_male_adult",    "label": "古风男声",
     "description": "中年男声，音色低沉古朴带苍凉感。武侠小说里中年江湖侠客的腔调——语速沉稳缓慢、吐字带古韵和拖音、重音常落在仗义或感慨的关键词上，偶有沉重的叹息感。要演绎出戏剧化的江湖沧桑，不是现代普通话的标准发音。"},
    {"tag": "philosopher_male_adult", "label": "哲思男声",
     "description": "中年男声，音色温润内敛。古装剧里道家高人或佛门长老的讲经腔——语速极慢、每个字之间都有思考停顿，重音落在哲理的关键词上，气息带禅意。要演绎出戏剧化的禅思感，不是日常对话的随意。"},
    {"tag": "ancient_female_youth",  "label": "古风女声",
     "description": "二十出头的青年女声，音色清丽飘逸带古典韵味。仙侠剧里上古仙子或古装剧大家闺秀的腔调——语调温婉缓慢、吐字软糯有古韵、句尾常带绵长拖音。要演绎出戏剧化的飘逸古典感，不是现代普通话的标准发音。"},
    {"tag": "fierce_female_elder",   "label": "强势女声",
     "description": "五十岁以上的老年女声，嗓音威严厚重略带沙哑。宫斗剧太皇太后或武侠片武林老前辈的腔调——语调铿锵有力、重音落在权威词上、不怒自威，整体气场压人。要演绎出戏剧化的威严感，不是日常老年人闲聊的平淡。"},

    # ---- 童声 / 故事 ----
    # 童声中性，按性格区分（lively / gentle）。男女物理特征在童年期重叠，
    # 故不再用 child_male / child_female 这种性别命名。
    {"tag": "child_lively",          "label": "活泼小孩",
     "description": "六七岁的小孩声线（中性），音色清脆响亮带奶音。儿童动画里调皮主角的腔调——语调起伏夸张、句尾上扬带兴奋感、重音故意夸张、整体充满好奇心和小淘气劲头。要演绎出戏剧化的童真活泼感，像动画片里冒险小英雄那样充满精力，不是真实小朋友的随便说话。"},
    {"tag": "child_gentle",          "label": "安静小孩",
     "description": "六七岁的小孩声线（中性），音色清细柔软带奶音。儿童剧或绘本里乖巧主角的腔调——语速偏慢、语调轻柔、句尾常带轻柔的拖音、重音落在害羞或好奇的词上。要演绎出戏剧化的童真乖巧感，像故事里那种安静却聪明的小孩，不是真实小朋友的日常聊天。"},
    {"tag": "child_timid",           "label": "胆小小孩",
     "description": "六七岁的小孩声线（中性），音色清细带奶音、气息略弱。儿童剧里被欺负的胆小角色或受惊吓的小主角的腔调——音量偏小、句尾常带颤抖或下沉、重音从不落在自己身上、整体试探性十足。要演绎出戏剧化的弱小怕事感，像动画片里被妈妈藏在身后的胆怯小孩，不是真实小朋友的日常内向。"},
    {"tag": "child_cheerful",        "label": "阳光小孩",
     "description": "六七岁的小孩声线（中性），音色清亮温暖带奶音、笑意盈盈。家庭剧里温暖善良的小主角或绘本里乐于助人的小天使的腔调——语调上扬带笑、重音落在表关心或快乐的词上、整体阳光善良不调皮。要演绎出戏剧化的童真治愈感，像广告片里那种甜笑的小孩，不是真实小朋友的随意。"},
    {"tag": "child_clever",          "label": "鬼灵精怪",
     "description": "六七岁的小孩声线（中性），音色清脆灵动带奶音。喜剧片或动画里小机灵主角的腔调——语速明快、语调起伏夸张带俏皮、重音故意落在出人意料的位置制造逗趣感、句尾常带得意的小拖音。要演绎出戏剧化的鬼精灵感，像动画里爱出小主意捉弄人的小聪明，不是真实小朋友的随便。"},
    {"tag": "child_melancholy",      "label": "孤独小孩",
     "description": "六七岁的小孩声线（中性），音色清细柔软带奶音、气息略带疲倦。文艺片或悲剧里孤儿或失意小主角的腔调——语速慢、语调起伏小但每个字都带感伤的拖音、句尾常下沉、整体藏着不应属于这个年纪的沉重。要演绎出戏剧化的童年忧伤感，像电影里那种孤独站在窗边的小孩，不是真实小朋友的日常情绪。"},
    {"tag": "storybook_female",      "label": "绘本女声",
     "description": "三十岁左右的成年女声，音色柔软温暖。专业绘本朗读那种带表演感的讲述腔——语调起伏明显、重音落在动作和拟声词上、不同情节切换不同情绪，像在用声音演角色给孩子看绘本。要演绎出戏剧化的故事表演感，不是平铺直叙的朗读。"},
    {"tag": "kidstory_female",       "label": "少儿故事女",
     "description": "三十岁左右的女声，嗓音温暖亲切带童趣。儿童动画片旁白阿姨的腔调——比成年女声更带明朗感、语调上扬带哄孩子的温柔、重音落在让小朋友兴奋的词上、句尾常带轻快的拖音。要演绎出戏剧化的儿童亲切感，不是给成年人听的平铺。"},

    # ---- 怯懦 / 忧郁 ----
    {"tag": "timid_male_youth",      "label": "怯懦少年男",
     "description": "二十出头的青年男声，音量偏小、气息略弱、句尾下沉带轻微吞音。校园剧里被欺负的胆小男配角——戏剧化的怯懦腔，重音从不落在自己身上、常用试探性的小停顿。要演绎出让人心疼的弱者感，不是真实日常的内向。"},
    {"tag": "timid_female_youth",    "label": "怯懦少女",
     "description": "二十出头的青年女声，音色清细、气声明显、句尾拖软下沉。校园剧或言情剧里被欺负的胆小女配——戏剧化的怯懦腔，重音从不落在自己身上、说话像在试探对方反应。要演绎出让人心疼的弱者感，不是日常的内向腼腆。"},
    {"tag": "melancholy_male_youth", "label": "忧郁青年男",
     "description": "二十多岁的青年男声，中低音区为主，音色干净带淡淡疲倦感。文艺片或悲剧里忧郁男主角的腔调——语速慢、句间停顿长，气息略带叹息，重音落在感伤词上像在自言自语。要演绎出戏剧化的内敛忧郁，不是日常吐槽的平淡。"},
    {"tag": "melancholy_female_youth", "label": "忧郁女青年",
     "description": "二十多岁的青年女声，中音偏暗、音色柔和略带气声。文艺片或悲剧里忧郁女主角的腔调——语速慢、语调起伏小但每个字都带感伤的拖音、气息略带叹息。要演绎出戏剧化的内敛忧伤感，像心里藏了一段故事的人，不是日常情绪平淡的女生。"},

    # ---- 补 cell 空缺：成年男热血/温暖、女青年冷酷、成年女心机/果敢 ----
    {"tag": "passionate_male_adult", "label": "热血中年男",
     "description": "三十多岁的成年男声，中音区为主、气息充沛、声音洪亮。武侠片里中年大侠或战争片里指挥官的腔调——重音夸张外放、语调起伏大、关键句子高亢有力，能燃起人热血。要演绎出戏剧化的号召力和燃感，不是日常对话的平静。"},
    {"tag": "kind_male_adult",       "label": "温暖父亲男",
     "description": "三十多岁的成年男声，音色温厚柔和、胸腔共鸣足。家庭剧里温暖父亲的腔调——语速平缓带笑意，重音落在表关爱的词上，句尾偶有放轻的拖音，像给孩子讲故事那种暖。要演绎出戏剧化的父爱感，不是同事间的客套寒暄。"},
    {"tag": "cold_female_youth",     "label": "高冷少女",
     "description": "二十出头的青年女声，音色清冷干净、气息克制不带笑意。校园剧里高傲不易接近的学姐或古偶剧冷艳女配的腔调——语调下沉平直、说话简短利落、句尾干脆收尾，整体疏离带威压。要演绎出戏剧化的高冷感，不是真实的内向腼腆。"},
    {"tag": "cunning_female_adult",  "label": "心机姐姐",
     "description": "三十多岁的成年女声，中音偏低、音色柔润。宫斗剧运筹帷幄的妃嫔或职场剧笑里藏刀的姐姐的腔调——语速缓、咬字清晰、句间长停顿，重音故意压低让话里有话，整体温柔包装下的精明冷感。要演绎出戏剧化的心机感，不是日常聊天的随意。"},
    {"tag": "brave_female_adult",    "label": "女将军",
     "description": "三十岁左右的成年女声，中音偏亮、气息有力。武侠片独当一面的女侠或战争剧女军官的腔调——发音干脆有节奏、语调坚定有决心、重音夸张落在号令关键词上，整体果敢凛然。要演绎出戏剧化的领导力和侠气，不是日常对话的平淡。"},

    # ---- 老年扩充：补 4 男 + 4 女（从 wise_male_elder + fierce_female_elder 的 2 条扩到 8 条）----
    {"tag": "kind_male_elder",       "label": "慈祥老爷爷",
     "description": "六十多岁的老年男声，嗓音温厚柔和带轻微沙哑。家庭剧里慈祥外公或武侠片里隐居老者的腔调——语速缓慢、笑意盈盈、重音落在表关爱的词上、句尾常带温柔的拖音。要演绎出戏剧化的爷爷感和岁月温柔，不是普通老年人的日常碎语。"},
    {"tag": "fierce_male_elder",     "label": "严厉老前辈",
     "description": "六十多岁的老年男声，嗓音威严厚重略带沙哑。武侠片武林老前辈或家庭剧严厉爷爷的腔调——语调铿锵有力、重音故意压低制造威压、句尾干脆下沉，整体不怒自威让人不敢造次。要演绎出戏剧化的严厉长辈感，不是普通老人发火。"},
    {"tag": "cunning_male_elder",    "label": "老谋深算",
     "description": "六十多岁的老年男声，嗓音低沉带沙哑、气息略缓。武侠或宫斗剧里老狐狸/老谋臣的腔调——语速极慢、咬字清晰带停顿、重音故意压低让话里有话，整体笑里藏话的精明感。要演绎出戏剧化的老狐狸感，话每句都有第二层意思，不是普通老人的家常絮叨。"},
    {"tag": "wise_female_elder",     "label": "智慧老奶奶",
     "description": "六十多岁的老年女声，嗓音温润带岁月沙哑感。古装剧里得道女师太或现代剧资深女学者的腔调——语速沉稳、吐字考究、重音落在哲思关键词上、句间常有思考停顿。要演绎出戏剧化的睿智长者感，看尽世事的从容，不是普通老太太闲聊。"},
    {"tag": "kind_female_elder",     "label": "慈祥老奶奶",
     "description": "六十多岁的老年女声，嗓音温暖柔和带轻微沙哑。家庭剧里慈祥外婆或绘本里讲故事的奶奶的腔调——语速缓慢、笑意明显、重音落在表疼爱的词上、句尾常带温柔的拖音。要演绎出戏剧化的奶奶感和母性长者光辉，不是日常老人的随便。"},
    {"tag": "melancholy_female_elder", "label": "沧桑老妇人",
     "description": "六十多岁的老年女声，嗓音柔和略沙哑、带淡淡疲倦感。年代剧或文艺片里历经沧桑的老妇人的腔调——语速极慢、句间停顿长、气息常带轻轻的叹息、重音落在感伤词上像在回忆往事。要演绎出戏剧化的岁月沧桑感，藏着大半辈子故事的那种内敛悲悯，不是普通老人的悲叹。"},
]


def _check_tags_match() -> list[str]:
    """Return tags in VD_CATALOG that don't have a TAG_TO_ATTRS mapping —
    those would be silently dropped from speakers.json without this check."""
    return [e["tag"] for e in VD_CATALOG if e["tag"] not in TAG_TO_ATTRS]


def synth_one(model: Module, description: str, sample_text: str) -> bytes:
    """Run one VoiceDesign generation. Returns 24 kHz mono 16-bit WAV bytes.

    **Must return WAV, not AAC**: prompts are written to disk as ``.wav``
    and re-loaded at TTS runtime by the model's audio loader, which
    expects PCM/WAV. (An earlier version of this script accidentally
    saved AAC bytes under .wav extension after the runtime path was
    refactored to AAC, breaking every prompt at load time. See the
    ``_collect_audio_wav`` vs ``_collect_audio`` split in
    ``app/services/tts_qwen3.py``.)
    """
    generator = model.generate_voice_design(
        text=sample_text,
        instruct=description,
        language="Chinese",
        verbose=False,
    )
    return _collect_audio_wav(generator)


def build_vd_speakers(catalog: list[dict[str, str]], now_iso: str) -> list[Speaker]:
    """Translate VD_CATALOG entries into Speaker rows. Skips tags that
    have no TAG_TO_ATTRS mapping (caller already warned)."""
    out: list[Speaker] = []
    for entry in catalog:
        attrs = TAG_TO_ATTRS.get(entry["tag"])
        if attrs is None:
            continue
        gender, age, personality = attrs
        out.append(Speaker(
            speaker_id=f"zs:vd_{entry['tag']}",
            gender=gender,
            age=age,
            personality=list(personality),
            source="qwen3-tts-voicedesign",
            notes=f"{entry['label']} | generated {now_iso}",
        ))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-dir", required=True,
        help="Path to a Qwen3-TTS VoiceDesign model directory.",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Override prompts dir (default: settings.qwen3_prompts_dir).",
    )
    parser.add_argument(
        "--ids", default=None,
        help="Comma-separated subset of catalog tags (default: all).",
    )
    parser.add_argument(
        "--regenerate", action="store_true",
        help="Overwrite existing wav/txt files (default: skip cached).",
    )
    parser.add_argument(
        "--no-update-speakers", action="store_true",
        help="Skip the speakers.json merge (just write WAVs).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    settings = get_settings()

    out_dir = Path(args.out_dir) if args.out_dir else settings.qwen3_prompts_dir
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sanity-check tag coverage before doing expensive model load.
    missing = _check_tags_match()
    if missing:
        log.warning(
            "%d catalog tag(s) have no TAG_TO_ATTRS mapping (will be "
            "skipped from speakers.json): %s",
            len(missing), ", ".join(missing),
        )

    catalog = VD_CATALOG
    if args.ids:
        wanted = {t.strip() for t in args.ids.split(",") if t.strip()}
        catalog = [e for e in VD_CATALOG if e["tag"] in wanted]
        unknown = wanted - {e["tag"] for e in VD_CATALOG}
        if unknown:
            log.warning("--ids: unknown tag(s) ignored: %s", ", ".join(unknown))
        if not catalog:
            log.error("--ids: no matching catalog entries; nothing to do")
            return 2

    # Heavy: load the VoiceDesign model. Validate type so a wrong-variant
    # download fails loud rather than at first generation.
    log.info("loading model from %s", args.model_dir)
    try:
        from mlx_audio.tts.utils import load_model
    except ImportError as exc:
        log.error("mlx-audio not installed: %s", exc)
        return 3
    model_path = Path(args.model_dir).resolve()
    if not (model_path / "config.json").exists():
        log.error(
            "model not found at %s (config.json missing).\n"
            "Either the dir is empty or the snapshot_download didn't "
            "fetch a real repo. The current working VoiceDesign repo is\n"
            "  mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit\n"
            "See this script's docstring for the full download command.",
            model_path,
        )
        return 4
    t0 = time.monotonic()
    model = load_model(model_path)
    model_type = getattr(model.config, "tts_model_type", None)
    if model_type != "voice_design":
        log.error(
            "model at %s has tts_model_type=%r; need 'voice_design'. "
            "Download a VoiceDesign variant (see script docstring).",
            args.model_dir, model_type,
        )
        return 4
    log.info("model loaded in %.1fs (type=%s)", time.monotonic() - t0, model_type)

    success, failures, skipped = 0, [], 0

    for entry in catalog:
        tag = entry["tag"]
        prompt_id = f"vd_{tag}"
        wav_path = out_dir / f"{prompt_id}.wav"
        txt_path = out_dir / f"{prompt_id}.txt"

        if wav_path.exists() and not args.regenerate:
            log.info("skip %s (cached; pass --regenerate to redo)", prompt_id)
            skipped += 1
            continue

        log.info("synth %s ← %s", prompt_id, entry["label"])
        try:
            t0 = time.monotonic()
            sample_text = sample_text_for(tag)
            wav_bytes = synth_one(model, entry["description"], sample_text)
            wav_path.write_bytes(wav_bytes)
            txt_path.write_text(sample_text, encoding="utf-8")
            log.info("  ok in %.1fs (%.1f KB)",
                     time.monotonic() - t0, len(wav_bytes) / 1024)
            success += 1
        except Exception as exc:
            log.error("  failed: %s", exc)
            failures.append((prompt_id, str(exc)))

    log.info("---")
    log.info("done: %d new wavs, %d cached, %d failed in %s",
             success, skipped, len(failures), out_dir)

    if not args.no_update_speakers:
        timestamp = datetime.now(timezone.utc).isoformat()
        # Build entries only for tags whose WAVs actually exist on disk
        # — a half-finished run shouldn't register speakers we can't
        # serve.
        present_tags = {
            e["tag"] for e in catalog
            if (out_dir / f"vd_{e['tag']}.wav").exists()
        }
        catalog_for_merge = [e for e in catalog if e["tag"] in present_tags]
        new_speakers = build_vd_speakers(catalog_for_merge, timestamp)
        speakers_path = settings.voice_library_path
        merged = merge_library(speakers_path, new_speakers, clean=False)
        write_library(speakers_path, merged)
        log.info("speakers.json: wrote %d total entries (%d new vd_*) → %s",
                 len(merged), len(new_speakers), speakers_path)

    if failures:
        log.warning("failures:")
        for pid, reason in failures:
            log.warning("  %s: %s", pid, reason)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
