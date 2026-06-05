//! Application state: process selection, the editable parameter form, and the
//! generated series fed to the chart.

use ratatui::widgets::ListState;

use crate::registry::NamedSeries;
use crate::registry::ParamKind;
use crate::registry::ParamValue;
use crate::registry::ParamValues;
use crate::registry::ProcessDescriptor;
use crate::registry::registry;

/// Which pane currently receives key input.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Focus {
    List,
    Form,
}

/// A form row: either a constructor parameter or the path-count control.
pub enum FieldRole {
    Param {
        name: &'static str,
        kind: ParamKind,
        doc: &'static str,
    },
    Paths,
}

/// One editable form row with its text buffer.
pub struct Field {
    pub role: FieldRole,
    pub buffer: String,
}

impl Field {
    pub fn label(&self) -> &str {
        match &self.role {
            FieldRole::Param { name, .. } => name,
            FieldRole::Paths => "paths (M)",
        }
    }

    pub fn hint(&self) -> &'static str {
        match &self.role {
            FieldRole::Param { kind, .. } => kind.hint(),
            FieldRole::Paths => "uint",
        }
    }

    pub fn doc(&self) -> &str {
        match &self.role {
            FieldRole::Param { doc, .. } => doc,
            FieldRole::Paths => "Number of Monte-Carlo paths to overlay",
        }
    }
}

pub struct App {
    pub processes: Vec<&'static ProcessDescriptor>,
    pub list_state: ListState,
    pub fields: Vec<Field>,
    pub field_idx: usize,
    pub focus: Focus,
    pub filter: String,
    pub filtering: bool,
    pub series: Vec<NamedSeries>,
    pub status: String,
    pub should_quit: bool,
}

impl App {
    pub fn new() -> Self {
        let processes = registry();
        let mut list_state = ListState::default();
        if !processes.is_empty() {
            list_state.select(Some(0));
        }
        let mut app = Self {
            processes,
            list_state,
            fields: Vec::new(),
            field_idx: 0,
            focus: Focus::List,
            filter: String::new(),
            filtering: false,
            series: Vec::new(),
            status: "Select a process · ⏎/g generate · Tab switch pane · q quit".to_string(),
            should_quit: false,
        };
        app.rebuild_fields();
        app
    }

    /// Descriptors matching the current filter, in registry order.
    pub fn visible(&self) -> Vec<&'static ProcessDescriptor> {
        let f = self.filter.to_lowercase();
        self.processes
            .iter()
            .copied()
            .filter(|d| {
                f.is_empty()
                    || d.name.to_lowercase().contains(&f)
                    || d.category.label().to_lowercase().contains(&f)
            })
            .collect()
    }

    pub fn selected(&self) -> Option<&'static ProcessDescriptor> {
        let visible = self.visible();
        let idx = self.list_state.selected()?;
        visible.get(idx).copied()
    }

    /// Rebuild the form from the selected process's parameter defaults.
    pub fn rebuild_fields(&mut self) {
        let mut fields = Vec::new();
        if let Some(desc) = self.selected() {
            for spec in desc.params {
                let buffer = match spec.default {
                    crate::registry::ParamDefault::F64(v) => format_f64(v),
                    crate::registry::ParamDefault::Usize(v) => v.to_string(),
                    crate::registry::ParamDefault::OptF64(v) => opt_to_string(v, format_f64),
                    crate::registry::ParamDefault::OptBool(v) => opt_to_string(v, |b| b.to_string()),
                };
                fields.push(Field {
                    role: FieldRole::Param {
                        name: spec.name,
                        kind: spec.kind,
                        doc: spec.doc,
                    },
                    buffer,
                });
            }
        }
        fields.push(Field {
            role: FieldRole::Paths,
            buffer: "10".to_string(),
        });
        self.fields = fields;
        self.field_idx = 0;
    }

    /// Parse the form and generate paths, replacing the chart series.
    pub fn generate(&mut self) {
        let Some(desc) = self.selected() else {
            self.status = "No process selected".to_string();
            return;
        };
        let mut values = ParamValues::default();
        let mut paths = 1usize;
        for field in &self.fields {
            match &field.role {
                FieldRole::Param { name, kind, .. } => match parse_param(*kind, &field.buffer) {
                    Ok(value) => values.set(name, value),
                    Err(msg) => {
                        self.status = format!("'{name}': {msg}");
                        return;
                    }
                },
                FieldRole::Paths => match field.buffer.trim().parse::<usize>() {
                    Ok(m) if m >= 1 => paths = m,
                    _ => {
                        self.status = "'paths (M)': expected a positive integer".to_string();
                        return;
                    }
                },
            }
        }

        let source = (desc.build)(&values);
        let samples = source.sample_par(paths);
        self.series = samples.into_iter().flatten().collect();
        self.status = format!(
            "Generated {paths} path(s) of {} · {} series",
            desc.name,
            self.series.len()
        );
    }

    /// Data bounds across all generated series, padded for display.
    pub fn bounds(&self) -> ([f64; 2], [f64; 2]) {
        if self.series.is_empty() {
            return ([0.0, 1.0], [0.0, 1.0]);
        }
        let mut x_max = 0.0f64;
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;
        for s in &self.series {
            for &(x, y) in &s.points {
                x_max = x_max.max(x);
                if y.is_finite() {
                    y_min = y_min.min(y);
                    y_max = y_max.max(y);
                }
            }
        }
        if !y_min.is_finite() || !y_max.is_finite() {
            y_min = 0.0;
            y_max = 1.0;
        }
        let pad = ((y_max - y_min) * 0.05).max(1e-9);
        ([0.0, x_max.max(1.0)], [y_min - pad, y_max + pad])
    }
}

fn parse_param(kind: ParamKind, raw: &str) -> Result<ParamValue, String> {
    let s = raw.trim();
    match kind {
        ParamKind::F64 => s
            .parse::<f64>()
            .map(ParamValue::F64)
            .map_err(|_| "expected a number".to_string()),
        ParamKind::Usize => s
            .parse::<usize>()
            .map(ParamValue::Usize)
            .map_err(|_| "expected a non-negative integer".to_string()),
        ParamKind::OptF64 => {
            if is_none(s) {
                Ok(ParamValue::OptF64(None))
            } else {
                s.parse::<f64>()
                    .map(|v| ParamValue::OptF64(Some(v)))
                    .map_err(|_| "expected a number or 'none'".to_string())
            }
        }
        ParamKind::OptBool => {
            if is_none(s) {
                Ok(ParamValue::OptBool(None))
            } else {
                parse_bool(s)
                    .map(|b| ParamValue::OptBool(Some(b)))
                    .ok_or_else(|| "expected true/false or 'none'".to_string())
            }
        }
    }
}

fn is_none(s: &str) -> bool {
    s.is_empty() || s.eq_ignore_ascii_case("none") || s == "-"
}

fn parse_bool(s: &str) -> Option<bool> {
    match s.to_lowercase().as_str() {
        "true" | "t" | "1" | "yes" | "y" => Some(true),
        "false" | "f" | "0" | "no" | "n" => Some(false),
        _ => None,
    }
}

fn format_f64(v: f64) -> String {
    let s = format!("{v}");
    s
}

fn opt_to_string<T>(v: Option<T>, f: impl Fn(T) -> String) -> String {
    match v {
        Some(value) => f(value),
        None => "none".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::event::handle_key;
    use crossterm::event::KeyCode;
    use crossterm::event::KeyEvent;

    #[test]
    fn registry_is_populated_and_form_built() {
        let app = App::new();
        assert!(!app.processes.is_empty());
        assert!(app.fields.len() >= 2);
        assert!(matches!(app.fields.last().unwrap().role, FieldRole::Paths));
    }

    #[test]
    fn generate_uses_default_path_count() {
        let mut app = App::new();
        app.generate();
        assert_eq!(app.series.len(), 10);
        assert!(app.series[0].points.len() > 100);
    }

    #[test]
    fn paths_field_controls_series_count() {
        let mut app = App::new();
        let last = app.fields.len() - 1;
        app.fields[last].buffer = "3".to_string();
        app.generate();
        assert_eq!(app.series.len(), 3);
    }

    #[test]
    fn invalid_parameter_is_reported_and_aborts() {
        let mut app = App::new();
        app.fields[0].buffer = "not-a-number".to_string();
        app.generate();
        assert!(app.series.is_empty());
        assert!(app.status.contains("expected"));
    }

    #[test]
    fn arrow_key_moves_selection_and_rebuilds_form() {
        let mut app = App::new();
        let first = app.selected().unwrap().name;
        handle_key(&mut app, KeyEvent::from(KeyCode::Down));
        let second = app.selected().unwrap().name;
        assert_ne!(first, second);
    }

    #[test]
    fn renders_empty_and_generated_without_panic() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let mut app = App::new();
        let mut terminal = Terminal::new(TestBackend::new(120, 40)).unwrap();
        terminal
            .draw(|frame| crate::app::ui::draw(frame, &mut app))
            .unwrap();
        app.generate();
        terminal
            .draw(|frame| crate::app::ui::draw(frame, &mut app))
            .unwrap();
        assert!(!app.series.is_empty());
    }
}
