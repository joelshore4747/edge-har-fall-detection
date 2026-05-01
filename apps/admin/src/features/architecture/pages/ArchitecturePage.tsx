import { Eyebrow } from '../../../components/v6'
import { D01SystemArchitecture } from '../diagrams/D01SystemArchitecture'
import { D02Pipeline } from '../diagrams/D02Pipeline'
import { RuntimeInferenceSequenceDiagram } from '../diagrams/RuntimeInferenceSequenceDiagram'
import { D04ERDiagram } from '../diagrams/D04ERDiagram'
import { D05SchemaMapping } from '../diagrams/D05SchemaMapping'
import { D06NavigationIA } from '../diagrams/D06NavigationIA'
import { D07FallFusion } from '../diagrams/D07FallFusion'

export function ArchitecturePage() {
  return (
    <>
      <div class="architecture-page">
        <Eyebrow>Architecture · seven figures</Eyebrow>

        <h1
          style={{
            'font-family': 'var(--serif)',
            'font-size': 'clamp(56px, 7vw, 96px)',
            'letter-spacing': '-.03em',
            'line-height': 0.95,
            'margin-bottom': '24px',
          }}
        >
          From a phone in a pocket<br />
          to a <em style={{ 'font-style': 'italic', color: 'var(--terracotta)' }}>row in PostgreSQL.</em>
        </h1>

        <div class="prose">
          <p>
            The system splits into four tiers: an edge client (Flutter app), the edge proxy (Nginx +
            Docker Compose), the runtime app (FastAPI + scikit-learn / XGBoost in-process), and the
            admin shells you are reading right now. The figures below use the deployed artifact and
            database names from this repository so each screenshot can stand as a dissertation figure.
          </p>
        </div>

        <D01SystemArchitecture />
        <D02Pipeline />
        <RuntimeInferenceSequenceDiagram />
        <D04ERDiagram />
        <D05SchemaMapping />
        <D06NavigationIA />
        <D07FallFusion />
      </div>
    </>
  )
}
