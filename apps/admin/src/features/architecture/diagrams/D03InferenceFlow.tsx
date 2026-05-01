import { DiagramFigure } from './DiagramFigure'

export function D03InferenceFlow() {
  return (
    <DiagramFigure
      num="03"
      title="Runtime inference"
      emphasis="flow"
      accent="var(--teal)"
      src="/figures/D03_runtime_inference_flow.svg"
      alt="UML sequence diagram of one inference round-trip"
      caption={
        <>
          <strong>Figure 03.</strong> UML sequence diagram of one inference round-trip. Lifelines
          run top-to-bottom; horizontal arrows are messages; the dashed teal column on{' '}
          <code>runtime_inference</code> is the activation bar during parallel HAR + Fall scoring.
          Wall-clock timestamps are read from the gutter on the left. Solid arrows are synchronous
          requests; dashed teal arrows are responses / completions.
        </>
      }
    />
  )
}
