"""Figure annotation schema and prompt used to construct Loom52M."""

import json

CONTENT_TYPE_ENUM = [
    "chart", "microscopy", "blot_gel", "schematic", "photo",
    "chemical_structure", "table", "map", "radiograph", "simulation", "other",
]

MODALITY_ENUM = [
    "bar_chart", "grouped_bar_chart", "stacked_bar_chart", "line_chart",
    "scatter_plot", "box_plot", "violin_plot", "heatmap", "grid_of_heatmaps",
    "pie_chart", "density_plot", "histogram", "histogram_overlay",
    "survival_curve", "volcano_plot", "forest_plot", "dot_plot", "radar_chart",
    "venn_diagram", "network_graph", "dendrogram", "flow_cytometry_plot",
    "raster_plot", "psth_plot", "spectrum_plot", "chromatogram",
    "fluorescence_microscopy", "confocal_microscopy", "immunofluorescence",
    "brightfield_microscopy", "phase_contrast_microscopy",
    "electron_microscopy_tem", "electron_microscopy_sem",
    "atomic_force_microscopy", "scanning_tunneling_microscopy",
    "live_cell_imaging", "optical_coherence_tomography",
    "histology_stain", "immunohistochemistry", "in_situ_hybridization",
    "western_blot", "northern_blot", "southern_blot", "gel_electrophoresis",
    "x_ray", "ct_scan", "mri", "pet_scan", "spect_scan", "ultrasound",
    "mammography", "angiography", "endoscopy", "fluoroscopy", "dermoscopy",
    "ecg_trace", "eeg_trace", "meg_trace", "emg_trace",
    "3d_structural_model", "molecular_structure", "crystal_structure",
    "molecular_dynamics_simulation",
    "pathway_diagram", "flow_diagram", "workflow_diagram",
    "anatomical_illustration", "phylogenetic_tree", "choropleth_map",
    "geographic_map", "clinical_photo", "illustration", "other",
]

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "regions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "region_id": {"type": "string"},
                    "content_type": {"type": "string", "enum": CONTENT_TYPE_ENUM},
                    "modality": {"type": "string", "enum": MODALITY_ENUM},
                    "box_2d": {
                        "type": "array", "items": {"type": "integer"},
                        "minItems": 4, "maxItems": 4,
                        "description": "[ymin,xmin,ymax,xmax] normalized 0-1000 relative to the FULL image, tightly bounding this panel",
                    },
                    "layout": {
                        "type": "object",
                        "properties": {"rows": {"type": "integer"}, "cols": {"type": "integer"}},
                        "required": ["rows", "cols"],
                        "description": "If this panel itself contains an internal grid of sub-plots/sub-images, give that grid's row/col count; otherwise {rows:1, cols:1}.",
                    },
                    "data_description": {
                        "type": "string",
                        "description": "1-2 sentences describing the DATA ITSELF, purely from what you see. Do NOT reference the article or interpret meaning here.",
                    },
                    "caption": {
                        "type": "string",
                        "description": "1-2 sentence caption fusing this panel's visual content with relevant context from the figure's original caption and article passages. Do not repeat data_description, do not copy source sentences verbatim.",
                    },
                },
                "required": ["region_id", "content_type", "modality", "box_2d", "layout", "data_description", "caption"],
            },
        }
    },
    "required": ["regions"],
}

def build_prompt(caption, mention_paragraphs):
    return f"""Segment this biomedical figure into its labeled sub-panels (e.g. A/B/C/D), or a single region if uniform.

Figure caption:
{caption}

Passages from the article body that discuss this figure:
{json.dumps(mention_paragraphs[:2], ensure_ascii=False)}

For each panel, give:
- region_id
- content_type and modality: pick the CLOSEST match from the fixed lists in the schema, do not invent new values
- box_2d: TIGHT box normalized 0-1000 relative to the full image
- layout: if this panel itself contains an internal grid of sub-plots/sub-images (small multiples), give {{rows, cols}}; otherwise {{rows:1, cols:1}}
- data_description: pure visual facts, no article context
- caption: interpretive, fusing visual + article context, only the parts relevant to THIS panel

data_description and caption must not overlap in content. Do not copy source sentences verbatim -- synthesize.
Return only the schema JSON."""
