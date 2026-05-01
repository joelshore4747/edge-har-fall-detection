import { DiagramFigure } from './DiagramFigure'

export function D06NavigationIA() {
  return (
    <DiagramFigure
      num="06"
      title="Navigation"
      emphasis="information architecture"
      accent="var(--ochre)"
      src="/figures/D06_navigation_ia.svg"
      alt="Two-shell information architecture sheet for the admin app"
      caption={
        <>
          <strong>Figure 06.</strong> Two-shell information architecture sheet. Both shells share
          the root path <code>/</code> but diverge by audience: the editorial shell carries the
          dissertation narrative and the seven figures, the analytics shell supports live operator
          monitoring and per-session review. Each route's purpose is captured in one line so the
          sheet doubles as a UX deliverable for stakeholders.
        </>
      }
    />
  )
}
