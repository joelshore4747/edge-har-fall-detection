import { DiagramFigure } from './DiagramFigure'

export function D01SystemArchitecture() {
  return (
    <DiagramFigure
      num="01"
      title="Full system"
      emphasis="architecture"
      accent="var(--terracotta)"
      src="/figures/D01_full_system_architecture.svg"
      alt="Container-level system architecture for UniFall Monitor"
      caption={
        <>
          <strong>Figure 01.</strong> Container-level architecture for UniFall Monitor. Top row
          shows the synchronous write path from a Flutter client to PostgreSQL through a single
          API tier; bottom row shows the read side (SolidJS admin shells) and the artifact load.
          Stroke style and colour both encode the channel, so the figure remains legible in
          grayscale.
        </>
      }
    />
  )
}
