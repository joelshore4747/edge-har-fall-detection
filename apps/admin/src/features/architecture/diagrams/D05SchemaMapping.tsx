import { DiagramFigure } from './DiagramFigure'

export function D05SchemaMapping() {
  return (
    <DiagramFigure
      num="05"
      title="Data schema"
      emphasis="mapping"
      accent="var(--teal)"
      src="/figures/D05_data_schema_mapping.svg"
      alt="Sankey column mapping of datasets to harmonised features and labels"
      caption={
        <>
          <strong>Figure 05.</strong> Dataset-to-feature harmonisation. Source corpora are mapped
          into a common per-window feature representation (time-domain, frequency, and
          magnitude-derived families) before branching into the 4-class HAR label space and the
          binary fall / non-fall space. Sankey ribbon width is proportional to subject count;
          centreline strokes preserve legibility in grayscale.
        </>
      }
    />
  )
}
