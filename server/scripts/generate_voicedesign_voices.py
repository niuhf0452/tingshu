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


# Per-tag reference text for VoiceDesign generation.
#
# Reference audio's CONTENT TYPE biases the clone's delivery style: a
# heavily-emoted reference produces a heavily-emoted clone, no matter
# what runtime tone you ask for. The whole catalog therefore aims at
# a calm,有声书-style delivery — short, neutral declarative sentences
# per tag (12-18 chars + 。), no quotes / exclamations / dramatic
# phrasing.
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
# produced 0/10 bleed. So this table gives each tag a unique short
# declarative sentence, distinct enough to not reinforce a single
# attention pattern across speakers.
#
# Age tier is preserved softly: child/teen tags get age-appropriate
# scenes (school, toys, play) for matched prosody, but the mood stays
# neutral throughout — no emotional outbursts, no dramatic phrasing.
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
    {"tag": "suspense_male_adult",   "label": "低沉沉稳男",
     "description": "中年男声，声线低沉略带气声，中低音区。语速偏慢，句间停顿自然，语调平直少起伏，吐字清晰但不刻意强调。整体安静沉着，像在安静的房间里慢慢说话，而不是在演戏。"},
    {"tag": "magnetic_male_adult",   "label": "温润中年男",
     "description": "中年男声，胸腔共鸣足，嗓音温润有质感。语速从容，气息平稳，句尾自然收住。说话亲切但克制，像深夜电台轻声讲述，不刻意营造磁性或亲密感。"},
    {"tag": "calm_male_adult",       "label": "沉稳中年男",
     "description": "中年男声，音色平实沉稳，中音区为主。语速均匀，语调起伏小，吐字清晰但不刻意强调。说话从容自然，没有戏剧性的重音和节奏感，像有声书里的平静讲述。"},
    {"tag": "wise_male_elder",       "label": "温厚长者男",
     "description": "五十多岁的老年男声，嗓音温厚带轻微沙哑。语速缓慢从容，句间偶有自然停顿，吐字清晰。说话平和带一点岁月感，像院子里和晚辈闲谈，不刻意装出深沉或哲思。"},
    {"tag": "cold_male_adult",       "label": "冷感男声",
     "description": "中年男声，声线冷峻干净，中低音区。语调平直少起伏，句尾自然下沉，吐字干脆。说话克制疏离，但不刻意制造威压感，整体安静而有距离。"},
    {"tag": "narrator_female_adult", "label": "流畅女旁白",
     "description": "三十岁左右的成年女声，中音偏暖，音色清润干净，不娇柔也不刻意稳重。讲故事的口吻，自然亲切，像姐姐在你耳边慢慢讲述，而不是在课堂上朗读课文。语调平稳少起伏，情绪克制，吐字清晰但不刻意强调，语速从容均匀，适合长篇小说类有声书的安静讲述。"},
    {"tag": "wise_female_adult",     "label": "知性女声",
     "description": "三十多岁的成年女声，中音偏低，嗓音温润有质感。语速沉稳，吐字考究但不刻意，整体带书卷气。说话内敛平和，像安静地分享所读的书，不刻意营造权威或思辨感。"},

    # ---- 男·少年/青年 ----
    {"tag": "youth_male_teen",       "label": "少年男声",
     "description": "十六七岁的少年男声，音色清亮带青涩。语调中等偏快，起伏自然，吐字清晰。声音里有少年特有的明快劲头，但说话克制，不喊也不刻意上挑。"},
    {"tag": "boyish_male_teen",      "label": "邻家男孩",
     "description": "十六七岁的少年男声，已过变声期，音色比成年男声明亮但具备男声厚度和胸腔共鸣。说话明快带笑意，语调自然上扬，整体亲切随和，像同班的好朋友在轻声聊天。"},
    {"tag": "cheerful_male_youth",   "label": "明亮青年男",
     "description": "二十岁出头的青年男声，声音明亮温暖，语调略带上扬。语速从容，吐字清晰，整体开朗自然。不刻意营造感染力或号召力，只是声音里自带阳光感。"},
    {"tag": "energetic_male_youth",  "label": "清爽青年男",
     "description": "二十岁出头的青年男声，音色清脆干净，语速偏明快。语调起伏自然，吐字利落，整体精神但不外放，像运动后微微喘着气在和人轻声说话。"},
    {"tag": "gentle_male_youth",     "label": "温柔青年男",
     "description": "二十多岁的青年男声，声线温柔细腻略带气声。语速缓慢，吐字软，说话自然亲切，不刻意拉长字音也不带哄人腔。"},
    {"tag": "warm_male_youth",       "label": "温暖男声",
     "description": "二十六七岁的青年男声，中音偏暖，嗓音宽厚有质感。语速从容，语调平稳，说话温和有耐心，比同龄人多一份成熟稳重，但不刻意表现温暖感。"},
    {"tag": "elegant_male_youth",    "label": "儒雅青年男",
     "description": "二十多岁的青年男声，音色清润儒雅，吐字工整。语速从容均匀，语调内敛少起伏，说话带书卷气而不刻意。"},
    {"tag": "fresh_male_youth",      "label": "清新男大学生",
     "description": "二十岁左右的男大学生声线，干净清爽，没有沧桑感。语速自然，语调略带上扬，说话明朗但不夸张，像大学校园里随意的对谈。"},

    # ---- 男·中年 ----
    {"tag": "arrogant_male_adult",   "label": "冷淡沉稳男",
     "description": "三十岁左右的成年男声，低沉富有质感，中低音区。语速从容，句尾偶尔轻轻下沉，吐字清晰。说话冷淡带一点距离感，但自然平和，不刻意营造压迫感。"},
    {"tag": "dub_male_adult",        "label": "工整男配",
     "description": "中年男声，吐字工整，语速沉稳，元音稍长，轻声字也咬清楚。整体带正式感但不刻意朗诵或拖音，像长辈一字一句地说话。"},
    {"tag": "podcast_male_adult",    "label": "深夜播客男",
     "description": "中年男声，嗓音温暖低沉，气息悠然。语速放慢，句间留白自然，说话亲切平和，像在深夜里轻声分享一段心事。"},
    {"tag": "lazy_male_youth",       "label": "慵懒青年男",
     "description": "二十多岁的青年男声，语调略慢，气息松弛，句尾自然带轻微拖音。说话随意自然，不带刻意的玩世不恭或痞气。"},

    # ---- 女·少女 / 青年 ----
    {"tag": "girl_female_teen",      "label": "邻家女孩",
     "description": "十五六岁少女声线，音色清亮带青涩。语调略带上扬，说话明快有节奏，整体鲜活自然，像放学路上和好朋友轻声说话。"},
    {"tag": "cute_female_teen",      "label": "清甜少女",
     "description": "十几岁的少女声线，音色清亮带些许鼻音。语调比成年人略高，起伏自然，句尾习惯轻轻上扬。说话明快带一点天真，清甜但不撒娇也不刻意拉长字音。"},
    {"tag": "sweet_female_youth",    "label": "甜美女青年",
     "description": "二十出头的青年女声，音色甜美悦耳带笑意。语调温柔，句尾自然收住，吐字软糯，说话亲切自然，不刻意拖音也不带告白腔。"},
    {"tag": "sweet_female_teen",     "label": "甜美少女",
     "description": "十六七岁少女声线，嗓音清甜柔软。语调起伏自然带青涩感，笑意盈盈，说话带少女特有的轻盈感，但不刻意营造羞涩拖音。"},
    {"tag": "fresh_female_youth",    "label": "清新女青年",
     "description": "二十多岁青年女声，音色清新干净。语速明快，语调略带上扬，吐字干净，说话自然不修饰，像清晨散步时的随口对谈。"},
    {"tag": "playful_female_youth",  "label": "俏皮女声",
     "description": "二十多岁青年女声，语气俏皮灵动。语调起伏自然，吐字明快，说话带一点活泼感但不刻意逗趣或挑逗。"},
    {"tag": "energetic_female_youth", "label": "明亮女青年",
     "description": "二十多岁青年女声，嗓音明亮，语速明快。语调起伏自然，吐字清晰利落，整体有精神但不外放，不带号召感。"},
    {"tag": "cheerful_female_adult", "label": "开朗姐姐",
     "description": "二十八九岁的成年女声，音色明亮温暖、笑意盈盈。语调自然带上扬，说话亲切自然，整体明朗但不刻意外放或夸张。"},
    {"tag": "gentle_female_youth",   "label": "温柔女青年",
     "description": "二十多岁青年女声，声线轻柔细腻略带气声。语速缓慢，吐字软糯，说话温柔自然，不刻意拉长字音。"},
    {"tag": "wise_female_youth",     "label": "知性女青年",
     "description": "二十多岁青年女声，音色温和稳重带书卷气。语调内敛，吐字考究但不刻意，说话沉静自然，像在图书馆里轻声交流。"},

    # ---- 女·成熟 / 御姐 ----
    {"tag": "cold_female_adult",     "label": "冷感御姐",
     "description": "三十岁左右的成熟女声，低沉冷艳带磁性。语调平稳缓慢，句尾自然下沉，气息克制。说话疏离但不刻意制造威压或杀气。"},
    {"tag": "tender_female_adult",   "label": "柔美女声",
     "description": "三十出头的成年女声，嗓音柔美绵软略带气声。语调缓慢，吐字软糯，说话温柔自然，不带刻意撒娇或亲密腔。"},
    {"tag": "charming_female_adult", "label": "低音女声",
     "description": "三十岁左右的成熟女声，音色低沉带气声。语速放慢，句尾自然带气声拖音，说话内敛沉静，不刻意营造性感或诱惑感。"},
    {"tag": "graceful_female_adult", "label": "温润淑女",
     "description": "三十岁左右的成熟女声，音色温润优雅。语速从容，吐字考究有韵味，说话端庄自然，不刻意表现仪态或贵气。"},
    {"tag": "mature_female_adult",   "label": "成熟女声",
     "description": "三十多岁的成熟女声，音色稳重大气。语调平稳，吐字干脆，说话从容有分寸，不带刻意威严或不容置疑感。"},
    {"tag": "kind_female_adult",     "label": "温柔妈妈",
     "description": "三十多岁的成熟女声，声音温暖宽厚带母性慈爱感。语速舒缓，吐字清晰有耐心，说话温和自然，不刻意表现关爱。"},

    # ---- 角色扮演 / 古风 ----
    {"tag": "noble_male_adult",      "label": "清贵男声",
     "description": "三十多岁的成年男声，音色清贵优雅。语调从容内敛，吐字考究有韵味，说话平和自然，不带刻意高傲或疏离感。"},
    {"tag": "ancient_male_adult",    "label": "古风男声",
     "description": "中年男声，音色低沉古朴，中低音区。语速沉稳缓慢，吐字带轻微古韵和拖音，说话平和有岁月感，不刻意制造苍凉或江湖感。"},
    {"tag": "philosopher_male_adult", "label": "禅意男声",
     "description": "中年男声，音色温润内敛。语速极慢，每字之间有自然停顿，气息平稳，说话沉静带一点禅意，不刻意装深沉或讲经。"},
    {"tag": "ancient_female_youth",  "label": "古风女声",
     "description": "二十出头的青年女声，音色清丽飘逸带古典韵味。语调温婉缓慢，吐字软糯有古韵，句尾自然收住，不刻意拖音。"},
    {"tag": "fierce_female_elder",   "label": "凛然老妇",
     "description": "五十岁以上的老年女声，嗓音厚重略带沙哑。语调铿锵稳健，吐字清晰，说话有威严感但不刻意张扬或压人。"},

    # ---- 童声 / 故事 ----
    # 童声中性，按性格区分（lively / gentle）。男女物理特征在童年期重叠，
    # 故不再用 child_male / child_female 这种性别命名。
    {"tag": "child_lively",          "label": "活泼小孩",
     "description": "六七岁的小孩声线（中性），音色清脆响亮带奶音。语调起伏自然，句尾略上扬，说话带好奇心和精神，但不刻意夸张或喊叫。"},
    {"tag": "child_gentle",          "label": "安静小孩",
     "description": "六七岁的小孩声线（中性），音色清细柔软带奶音。语速偏慢，语调轻柔，说话乖巧自然，带一点害羞但不刻意装乖。"},
    {"tag": "child_timid",           "label": "胆小小孩",
     "description": "六七岁的小孩声线（中性），音色清细带奶音、气息略弱。音量偏小，句尾自然下沉，说话试探性自然，不刻意装可怜或颤抖。"},
    {"tag": "child_cheerful",        "label": "阳光小孩",
     "description": "六七岁的小孩声线（中性），音色清亮温暖带奶音、笑意自然。语调略上扬，说话亲切善良，整体阳光自然，不刻意营造甜笑感。"},
    {"tag": "child_clever",          "label": "灵巧小孩",
     "description": "六七岁的小孩声线（中性），音色清脆灵动带奶音。语速明快，语调起伏自然带俏皮感，说话机灵但不刻意卖萌或装精怪。"},
    {"tag": "child_melancholy",      "label": "安静忧郁小孩",
     "description": "六七岁的小孩声线（中性），音色清细柔软带奶音、气息略缓。语速慢，语调起伏小，句尾自然下沉，说话带一点淡淡的安静感，不刻意拖音或装感伤。"},
    {"tag": "storybook_female",      "label": "绘本女声",
     "description": "三十岁左右的成年女声，音色柔软温暖。语调起伏自然，说话亲切带讲述感，像母亲在床边给孩子读绘本，但不刻意演角色。"},
    {"tag": "kidstory_female",       "label": "少儿故事女",
     "description": "三十岁左右的女声，嗓音温暖亲切带童趣。语调略上扬带温柔，说话明朗自然，亲切但不刻意哄孩子。"},

    # ---- 怯懦 / 忧郁 ----
    {"tag": "timid_male_youth",      "label": "怯懦少年男",
     "description": "二十出头的青年男声，音量偏小、气息略弱。句尾自然下沉，说话试探性自然，整体内敛，不刻意装怯懦或颤抖。"},
    {"tag": "timid_female_youth",    "label": "怯懦少女",
     "description": "二十出头的青年女声，音色清细、气声明显。句尾自然拖软，说话内敛带一点不安，但不刻意装弱或求关注。"},
    {"tag": "melancholy_male_youth", "label": "忧郁青年男",
     "description": "二十多岁的青年男声，中低音区为主，音色干净带淡淡疲倦感。语速慢，句间停顿自然，气息略带叹息，说话内敛沉静，像在自言自语。"},
    {"tag": "melancholy_female_youth", "label": "忧郁女青年",
     "description": "二十多岁的青年女声，中音偏暗、音色柔和略带气声。语速慢，语调起伏小，气息略带叹息，说话内敛沉静，不刻意拖音或装感伤。"},

    # ---- 补 cell 空缺 ----
    {"tag": "passionate_male_adult", "label": "明朗中年男",
     "description": "三十多岁的成年男声，中音区为主、气息充沛、声音明亮。语调起伏自然，吐字清晰有力，说话明朗有精神，但不刻意高亢或外放。"},
    {"tag": "kind_male_adult",       "label": "温暖父亲男",
     "description": "三十多岁的成年男声，音色温厚柔和、胸腔共鸣足。语速平缓带笑意，吐字清晰，说话温和有耐心，不刻意表现父爱或拖音。"},
    {"tag": "cold_female_youth",     "label": "冷感少女",
     "description": "二十出头的青年女声，音色清冷干净、气息克制。语调平直，说话简短利落，整体疏离但不刻意装高傲或制造威压。"},
    {"tag": "cunning_female_adult",  "label": "沉稳女声",
     "description": "三十多岁的成年女声，中音偏低、音色柔润。语速缓，吐字清晰，句间停顿自然，说话内敛沉稳，不刻意压低声音或制造话外音。"},
    {"tag": "brave_female_adult",    "label": "果敢女声",
     "description": "三十岁左右的成年女声，中音偏亮、气息有力。吐字干脆，语调坚定自然，说话果敢但不刻意张扬或喊号令。"},

    # ---- 老年扩充：补 4 男 + 4 女（从 wise_male_elder + fierce_female_elder 的 2 条扩到 8 条）----
    {"tag": "kind_male_elder",       "label": "慈祥老爷爷",
     "description": "六十多岁的老年男声，嗓音温厚柔和带轻微沙哑。语速缓慢带笑意，吐字清晰，说话温和自然，不刻意装慈祥或拖音。"},
    {"tag": "fierce_male_elder",     "label": "严肃老前辈",
     "description": "六十多岁的老年男声，嗓音厚重略带沙哑。语调稳健，吐字清晰，说话有威严感但不刻意压制或张扬。"},
    {"tag": "cunning_male_elder",    "label": "深沉老者",
     "description": "六十多岁的老年男声，嗓音低沉带沙哑、气息略缓。语速极慢，吐字清晰带停顿，说话内敛沉稳，不刻意装深沉或藏话外音。"},
    {"tag": "wise_female_elder",     "label": "智慧老奶奶",
     "description": "六十多岁的老年女声，嗓音温润带岁月沙哑感。语速沉稳，吐字考究，说话沉静自然，带从容感，不刻意营造哲思感。"},
    {"tag": "kind_female_elder",     "label": "慈祥老奶奶",
     "description": "六十多岁的老年女声，嗓音温暖柔和带轻微沙哑。语速缓慢，笑意自然，说话温和亲切，不刻意装慈爱或拖音。"},
    {"tag": "melancholy_female_elder", "label": "沉静老妇人",
     "description": "六十多岁的老年女声，嗓音柔和略沙哑、带淡淡疲倦感。语速极慢，句间停顿自然，气息带轻轻的叹息，说话内敛沉静，不刻意营造沧桑感。"},
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
