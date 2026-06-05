//! ratatui rendering: process list, parameter form, path chart, status bar.

use ratatui::Frame;
use ratatui::layout::Constraint;
use ratatui::layout::Layout;
use ratatui::layout::Rect;
use ratatui::style::Color;
use ratatui::style::Modifier;
use ratatui::style::Style;
use ratatui::symbols;
use ratatui::text::Line;
use ratatui::text::Span;
use ratatui::widgets::Axis;
use ratatui::widgets::Block;
use ratatui::widgets::Chart;
use ratatui::widgets::Dataset;
use ratatui::widgets::GraphType;
use ratatui::widgets::List;
use ratatui::widgets::ListItem;
use ratatui::widgets::Paragraph;

use crate::app::state::App;
use crate::app::state::ChartView;
use crate::app::state::Focus;

pub fn draw(frame: &mut Frame, app: &mut App) {
    let area = frame.area();
    let outer = Layout::vertical([Constraint::Min(0), Constraint::Length(1)]).split(area);
    // Fixed top height so the panes don't jump as processes with different
    // parameter counts are selected; long forms scroll within this area.
    let rows = Layout::vertical([Constraint::Length(16), Constraint::Min(6)]).split(outer[0]);
    let cols =
        Layout::horizontal([Constraint::Percentage(34), Constraint::Percentage(66)]).split(rows[0]);

    draw_list(frame, app, cols[0]);
    draw_form(frame, app, cols[1]);
    draw_chart(frame, app, rows[1]);
    draw_status(frame, app, outer[1]);
}

fn draw_list(frame: &mut Frame, app: &mut App, area: Rect) {
    let visible = app.visible();
    let items: Vec<ListItem> = visible
        .iter()
        .map(|d| {
            ListItem::new(Line::from(vec![
                Span::styled(format!("{:<15}", d.name), Style::default().fg(Color::White)),
                Span::styled(d.category.label(), Style::default().fg(Color::DarkGray)),
            ]))
        })
        .collect();

    let title = if app.filtering {
        format!(" Processes  /{}\u{2588} ", app.filter)
    } else if app.filter.is_empty() {
        format!(" Processes ({}) ", visible.len())
    } else {
        format!(" Processes (/{} \u{2192} {}) ", app.filter, visible.len())
    };

    let list = List::new(items)
        .block(
            Block::bordered()
                .title(title)
                .border_style(pane_style(app.focus == Focus::List)),
        )
        .highlight_style(
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("\u{25b8} ");

    frame.render_stateful_widget(list, area, &mut app.list_state);
}

fn draw_form(frame: &mut Frame, app: &App, area: Rect) {
    let name = app.selected().map(|d| d.name).unwrap_or("\u{2014}");
    let active = app.focus == Focus::Form;
    let mut lines: Vec<Line> = Vec::new();

    for (i, field) in app.fields.iter().enumerate() {
        let focused = active && i == app.field_idx;
        let marker = if focused { "\u{25b8}" } else { " " };
        let value_style = if focused {
            Style::default().fg(Color::Black).bg(Color::Yellow)
        } else {
            Style::default().fg(Color::Yellow)
        };
        let cursor = if focused { "\u{2588}" } else { " " };
        lines.push(Line::from(vec![
            Span::raw(format!("{marker} ")),
            Span::styled(
                format!("{:<11}", field.label()),
                Style::default().fg(Color::Gray),
            ),
            Span::styled(format!(" {}{cursor}", field.buffer), value_style),
            Span::styled(
                format!("  {}", field.hint()),
                Style::default().fg(Color::DarkGray),
            ),
        ]));
    }

    if let Some(field) = app.fields.get(app.field_idx) {
        lines.push(Line::raw(""));
        lines.push(Line::from(Span::styled(
            field.doc().to_string(),
            Style::default().fg(Color::DarkGray),
        )));
    }

    let inner_h = area.height.saturating_sub(2) as usize;
    let scroll = if active && app.field_idx + 2 >= inner_h {
        (app.field_idx + 2 - inner_h).min(lines.len().saturating_sub(inner_h)) as u16
    } else {
        0
    };
    let para = Paragraph::new(lines).scroll((scroll, 0)).block(
        Block::bordered()
            .title(format!(" Parameters \u{00b7} {name} "))
            .border_style(pane_style(active)),
    );
    frame.render_widget(para, area);
}

fn draw_chart(frame: &mut Frame, app: &App, area: Rect) {
    let ([x_min, x_max], [y_min, y_max]) = app.bounds();
    let total = app.series.len();
    let shown = app.shown_indices();
    let single = matches!(app.view, ChartView::Single(_));

    // Thin the dense path (n up to 1000) down to roughly the chart's Braille
    // resolution so segments don't pile into vertical smears.
    let target = (area.width as usize).saturating_mul(2).max(64);
    let prepared: Vec<(usize, Vec<(f64, f64)>)> = shown
        .iter()
        .map(|&k| (k, downsample(&app.series[k].points, target)))
        .collect();

    let datasets: Vec<Dataset> = prepared
        .iter()
        .map(|(k, points)| {
            let color = if single {
                Color::Rgb(125, 211, 255)
            } else {
                palette(*k, total)
            };
            let dataset = Dataset::default()
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(color))
                .data(points);
            if single || total <= 6 {
                dataset.name(app.series[*k].label.clone())
            } else {
                dataset
            }
        })
        .collect();

    let title = match app.view {
        _ if total == 0 => " Chart \u{00b7} press \u{23ce}/g to generate ".to_string(),
        ChartView::All => format!(" Chart \u{00b7} {total} paths \u{00b7} \u{2190}\u{2192} isolate "),
        ChartView::Single(i) => format!(
            " Chart \u{00b7} path {}/{} ({}) \u{00b7} \u{2190}\u{2192} switch ",
            i + 1,
            total,
            app.series.get(i).map(|s| s.label.as_str()).unwrap_or("")
        ),
    };

    let mid = (y_min + y_max) / 2.0;
    let chart = Chart::new(datasets)
        .block(Block::bordered().title(title))
        .x_axis(
            Axis::default()
                .style(Style::default().fg(Color::DarkGray))
                .bounds([x_min, x_max])
                .labels([Span::raw("0"), Span::raw(format!("{x_max:.0}"))]),
        )
        .y_axis(
            Axis::default()
                .style(Style::default().fg(Color::DarkGray))
                .bounds([y_min, y_max])
                .labels([
                    Span::raw(format!("{y_min:.2}")),
                    Span::raw(format!("{mid:.2}")),
                    Span::raw(format!("{y_max:.2}")),
                ]),
        );
    frame.render_widget(chart, area);
}

fn draw_status(frame: &mut Frame, app: &App, area: Rect) {
    let hint = if app.filtering {
        "filter: type to match \u{00b7} \u{23ce} apply \u{00b7} Esc clear"
    } else {
        match app.focus {
            Focus::List => "\u{2191}\u{2193} select \u{00b7} \u{23ce}/g generate \u{00b7} \u{2190}\u{2192} paths \u{00b7} / filter \u{00b7} Tab \u{2192} form \u{00b7} q quit",
            Focus::Form => "\u{2191}\u{2193} field \u{00b7} type to edit \u{00b7} \u{23ce}/g generate \u{00b7} \u{2190}\u{2192} paths \u{00b7} Tab/Esc \u{2192} list",
        }
    };
    let line = Line::from(vec![
        Span::styled(
            format!(" {} ", app.status),
            Style::default().fg(Color::Black).bg(Color::Cyan),
        ),
        Span::styled(format!("  {hint}"), Style::default().fg(Color::DarkGray)),
    ]);
    frame.render_widget(Paragraph::new(line), area);
}

/// Smooth cyan-to-violet gradient so an ensemble of paths reads as a coherent
/// fan rather than a clash of primary colours.
fn palette(i: usize, total: usize) -> Color {
    let t = if total <= 1 {
        0.0
    } else {
        i as f64 / (total - 1) as f64
    };
    let lerp = |a: f64, b: f64| (a + (b - a) * t) as u8;
    Color::Rgb(lerp(90.0, 200.0), lerp(210.0, 120.0), lerp(255.0, 250.0))
}

fn pane_style(active: bool) -> Style {
    if active {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default().fg(Color::DarkGray)
    }
}

/// Evenly thin `points` to about `target` samples, preserving the last point.
fn downsample(points: &[(f64, f64)], target: usize) -> Vec<(f64, f64)> {
    if points.len() <= target {
        return points.to_vec();
    }
    let step = points.len() as f64 / target as f64;
    let mut out = Vec::with_capacity(target + 2);
    let mut pos = 0.0;
    while (pos as usize) < points.len() {
        out.push(points[pos as usize]);
        pos += step;
    }
    let last = points[points.len() - 1];
    if out.last() != Some(&last) {
        out.push(last);
    }
    out
}
