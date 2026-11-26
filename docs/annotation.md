# Annotate and crop panels

BioLoom uses **Gemini 3.5 Flash-Lite** (`gemini-3.5-flash-lite`) for panel annotation. The script uses the Gemini Developer API. Create an [API key in Google AI Studio](https://aistudio.google.com/apikey) and set it in your shell.

Prepare an extracted image for each whole figure and create a JSON Lines file. Image paths are relative to the input file. Give each figure a stable, unique ID.

```json
{"id":"article1_fig2","image":"figures/figure2.jpg","caption":"Original figure caption","mention_paragraphs":["A body paragraph referring to Figure 2."]}
```

Pass the source caption and the actual mentioning paragraphs. Omit `mention_paragraphs` if none are available. The production prompt uses at most two passages. It asks for a visual description and a complementary caption grounded in the article context. It uses a fixed vocabulary for content type and modality.

```bash
export GEMINI_API_KEY="your-api-key"
python annotation/annotate.py figures.jsonl annotations.jsonl --workers 8
python annotation/crop.py figures.jsonl annotations.jsonl panels
```

The model identifier can be overridden with `--model`. Keep one writer per output file. Rerunning the annotation command skips completed IDs. The box format is `[ymin, xmin, ymax, xmax]` with integers in `[0, 1000]`. The crop script saves each panel with its parent ID and all annotation fields in `panels/panels.jsonl`.

The released Loom52M shards already contain whole-figure and panel training records. For the paper recipes, use those shards directly.
