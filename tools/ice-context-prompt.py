"""Single-call, interleaved context/key/ROI exploratory classification prompt."""

CONTEXT = '''Return a single JSON object, with context_photo_description as its first field.
Image 1 follows: the full photograph.
First describe what you can visibly identify across this photograph in context_photo_description:
water/ice patterns, sheets versus fragments, texture, lighting, ship/wake if visible,
and fog, wet-lens smears or other visibility limitations. Use a brief factual paragraph,
not a reasoning transcript. Distinguish observation from uncertainty. Describe the
whole photograph, including its foreground and background, rather than only a small patch.
This is exploratory assessment, not navigation or scientific measurement.'''

KEY = '''Image 2 follows: a reference key, not the target. Its rows contain three
human-approved examples per ice label. Use these project-specific visual categories:
- grease ice: a smooth or finely granular/slushy-looking surface, sometimes with slick
  streaks and subdued ripples; can be visually inseparable from smooth water.
- nilas: thin-looking dark/gray continuous film or sheets, sometimes with visible cracks,
  overlapping edges or flexible-looking broken plates.
- thin fyi: more opaque, established-looking sheet ice, often pale or snow-covered;
  an appearance label, not a measured thickness or confirmed age.
- icy bits: scattered small discrete ice fragments, with water visible between them.
- brash ice: concentrations of broken ice rubble and jumbled fragments.
- ice floe: larger coherent ice pieces or a broad intact ice surface filling the view.
These classes overlap physically; use the key to match the project's visual convention.
Prefer fragment/floe categories over inferred age when discrete pieces dominate.
Each key crop can include water/background; its label does not imply 100% coverage.
Do not classify the key, green text, borders, or title. Do not force grease ice when
smooth water is ambiguous; describe uncertainty and use unknown when unresolvable.'''

def build(baseline_prompt, images):
    """images arrive in old order: context, ROI, key; return ordered API blocks."""
    old = 'Image 1 shows context and an orange quadrilateral; image 2 is that exact extracted region. Assess ONLY that region.'
    assert old in baseline_prompt, 'Baseline image references changed; review prompt mapping'
    target = baseline_prompt.replace(old,
        'Image 3 follows: the exact extracted/rotated ROI from the orange outline in image 1. '
        'Classify ONLY image 3. Image 2 was the reference key, not the target. '
        'Use the full context photograph to interpret this region, but do not transfer ice outside '
        'the orange outline into its percentages. Keep context_photo_description about the whole of image 1. '
        'In a separate roi_description field, describe only the visible contents of image 3, '
        'including surface appearance and any visibility limitations. These are two distinct descriptions.')
    target = target.replace('Schema: {', 'Schema: {"context_photo_description":"brief factual description of the entire image 1, including foreground, background, visibility and uncertainty","roi_description":"brief factual description of only image 3, including surface appearance and visibility limitations",', 1)
    assert '"context_photo_description"' in target and '"roi_description"' in target
    target += '\nReturn context_photo_description first, roi_description second, then the visibility and classification fields. No text outside JSON.'
    texts = [CONTEXT, KEY, target]
    order = [0, 2, 1]
    blocks = []
    for text, index in zip(texts, order):
        blocks.extend([dict(type='text', content=text), dict(type='image', data_url=images[index])])
    return texts, blocks
