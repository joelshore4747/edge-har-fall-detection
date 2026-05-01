import { DiagramFigure } from './DiagramFigure'

export function D04ERDiagram() {
  return (
    <DiagramFigure
      num="04"
      title="Database"
      emphasis="entity relationships"
      accent="var(--ochre)"
      src="/figures/D04_database_er.svg"
      alt="Entity-relationship view of the deployed PostgreSQL schema"
      caption={
        <>
          <strong>Figure 04.</strong> Entity-relationship view of the deployed PostgreSQL schema —
          all 11 <code>app_*</code> tables, grouped into four bounded contexts. Foreign-key edges
          are drawn behind the cards and rendered with both crow's-foot glyphs and explicit
          cardinality badges so the figure remains legible in grayscale and at print resolution.
        </>
      }
    />
  )
}
