import type { JSXElement } from 'solid-js'
import { Section } from '../../../components/v6'

export function DiagramFigure(props: {
  num: string
  title: string
  emphasis?: string
  accent: string
  src: string
  alt: string
  caption: JSXElement
}) {
  return (
    <Section num={props.num} title={props.title} emphasis={props.emphasis}>
      <div class="fig figure-doc" style={{ '--accent-c': props.accent }}>
        <img class="fig-svg" src={props.src} alt={props.alt} loading="lazy" decoding="async" />
      </div>
      <div class="sci-cap">{props.caption}</div>
    </Section>
  )
}
