import { For, type JSXElement } from 'solid-js'

// Estimated character width for JetBrains Mono. Used so SVG text containers can
// be sized from the source string without measuring DOM. Tuned against the
// rendered font in the admin app's index.css.
const MONO_CHAR_W_AT_11PX = 6.6

export type LegendGlyph = 'solid' | 'dashed' | 'dotted' | 'arrow' | 'arrow-dashed'

export type LegendItem = {
  glyph: LegendGlyph
  colour: string
  label: string
}

/**
 * Bordered key block placed inside an SVG figure. Each row carries both a
 * line-style glyph and a colour swatch so the legend remains legible when the
 * figure is reproduced in grayscale.
 */
export function Legend(props: {
  x: number
  y: number
  title?: string
  items: readonly LegendItem[]
  width?: number
}) {
  const rowH = 20
  const padTop = props.title ? 30 : 14
  const padBottom = 12
  const w = props.width ?? 220
  const h = padTop + props.items.length * rowH + padBottom - 6
  return (
    <g transform={`translate(${props.x} ${props.y})`}>
      <rect x={0} y={0} width={w} height={h} rx={5} fill="var(--paper)" stroke="var(--line)" />
      {props.title && (
        <text x={12} y={18} class="lbl" fill="var(--text-3)">
          {props.title}
        </text>
      )}
      <For each={props.items}>
        {(item, i) => {
          const cy = padTop + i() * rowH + 2
          return (
            <g>
              <LegendGlyphMark x={12} y={cy} colour={item.colour} glyph={item.glyph} />
              <text
                x={62}
                y={cy + 4}
                font-family="var(--mono)"
                font-size="11"
                fill="var(--text-2)"
              >
                {item.label}
              </text>
            </g>
          )
        }}
      </For>
    </g>
  )
}

function LegendGlyphMark(props: { x: number; y: number; colour: string; glyph: LegendGlyph }) {
  const dash =
    props.glyph === 'dashed' || props.glyph === 'arrow-dashed'
      ? '6 4'
      : props.glyph === 'dotted'
        ? '2 3'
        : undefined
  const showArrow = props.glyph === 'arrow' || props.glyph === 'arrow-dashed'
  return (
    <g>
      <line
        x1={props.x}
        y1={props.y}
        x2={props.x + 38}
        y2={props.y}
        stroke={props.colour}
        stroke-width="2"
        stroke-dasharray={dash}
      />
      {showArrow && (
        <path
          d={`M${props.x + 38},${props.y - 4} L${props.x + 44},${props.y} L${props.x + 38},${props.y + 4} z`}
          fill={props.colour}
        />
      )}
    </g>
  )
}

/**
 * Single-line strip of expanded abbreviations. Lives at the bottom of the
 * figure so dissertation readers do not have to chase definitions in the
 * caption or surrounding prose.
 */
export function AbbrevRow(props: {
  x: number
  y: number
  width: number
  items: readonly string[]
}) {
  return (
    <g transform={`translate(${props.x} ${props.y})`}>
      <rect x={0} y={0} width={props.width} height={26} rx={4} fill="var(--paper)" stroke="var(--line)" />
      <text x={10} y={17} font-family="var(--mono)" font-size="10.5" fill="var(--text-3)">
        {props.items.join('  ·  ')}
      </text>
    </g>
  )
}

/**
 * Pill-shaped edge label that sizes itself from its text content rather than a
 * fixed width. Replaces the hardcoded 172px FlowLabel that overflowed long
 * strings.
 */
export function AutoLabel(props: {
  x: number
  y: number
  text: string
  colour: string
  fontSize?: number
}) {
  const fontSize = props.fontSize ?? 11.5
  const charW = MONO_CHAR_W_AT_11PX * (fontSize / 11)
  const padX = 12
  const w = Math.max(60, props.text.length * charW + padX * 2)
  return (
    <g>
      <rect
        x={props.x - w / 2}
        y={props.y - 13}
        width={w}
        height={26}
        rx={13}
        fill="var(--paper)"
        stroke={props.colour}
        stroke-width="1"
      />
      <text
        x={props.x}
        y={props.y + 4}
        text-anchor="middle"
        font-family="var(--mono)"
        font-size={String(fontSize)}
        fill={props.colour}
        font-weight="700"
      >
        {props.text}
      </text>
    </g>
  )
}

/**
 * Horizontal bar whose fill width is computed from a normalised value, with
 * the label on the left and the numeric value on the right. Replaces the
 * D07 gauge whose fill was a raw pixel width unrelated to the underlying
 * quantity.
 */
export function NormalisedBar(props: {
  x: number
  y: number
  width: number
  label: string
  value: number
  max?: number
  colour: string
  valueText?: string
  trackHeight?: number
}) {
  const max = props.max ?? 1
  const trackH = props.trackHeight ?? 10
  const ratio = Math.max(0, Math.min(1, props.value / max))
  const valueText = props.valueText ?? props.value.toFixed(2)
  return (
    <g>
      <text x={props.x} y={props.y} font-family="var(--mono)" font-size="11" fill="var(--text-2)">
        {props.label}
      </text>
      <text
        x={props.x + props.width}
        y={props.y}
        text-anchor="end"
        font-family="var(--mono)"
        font-size="11"
        fill="var(--text-3)"
      >
        {valueText}
      </text>
      <rect
        x={props.x}
        y={props.y + 6}
        width={props.width}
        height={trackH}
        rx={trackH / 2}
        fill="var(--bg-2)"
        stroke="var(--line)"
      />
      <rect
        x={props.x}
        y={props.y + 6}
        width={props.width * ratio}
        height={trackH}
        rx={trackH / 2}
        fill={props.colour}
      />
    </g>
  )
}

/**
 * Cardinality glyph used at the ends of D04 relationship lines. Adds an
 * explicit `1`, `1..N`, or `0..1` marker next to the crow's-foot so the
 * relationship reads in grayscale.
 */
export function Cardinality(props: { x: number; y: number; text: string; colour?: string }) {
  return (
    <g>
      <rect
        x={props.x - 14}
        y={props.y - 9}
        width={28}
        height={16}
        rx={3}
        fill="var(--paper)"
        stroke={props.colour ?? 'var(--text-3)'}
        stroke-width="0.8"
      />
      <text
        x={props.x}
        y={props.y + 3}
        text-anchor="middle"
        font-family="var(--mono)"
        font-size="9.5"
        fill={props.colour ?? 'var(--text-2)'}
        font-weight="700"
      >
        {props.text}
      </text>
    </g>
  )
}

export type SvgChildren = JSXElement
