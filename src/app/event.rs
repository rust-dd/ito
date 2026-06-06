//! Keyboard handling for the two panes and the filter prompt.

use crossterm::event::KeyCode;
use crossterm::event::KeyEvent;
use crossterm::event::KeyEventKind;
use crossterm::event::KeyModifiers;

use crate::app::state::App;
use crate::app::state::Focus;

pub fn handle_key(app: &mut App, key: KeyEvent) {
    if key.kind != KeyEventKind::Press {
        return;
    }
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        app.should_quit = true;
        return;
    }
    if app.filtering {
        handle_filter(app, key);
        return;
    }
    match app.focus {
        Focus::List => handle_list(app, key),
        Focus::Form => handle_form(app, key),
    }
}

fn handle_filter(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Esc => {
            app.filter.clear();
            app.filtering = false;
            reset_selection(app);
        }
        KeyCode::Enter => app.filtering = false,
        KeyCode::Backspace => {
            app.filter.pop();
            reset_selection(app);
        }
        KeyCode::Char(c) => {
            app.filter.push(c);
            reset_selection(app);
        }
        _ => {}
    }
}

fn handle_list(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Up | KeyCode::Char('k') => move_selection(app, -1),
        KeyCode::Down | KeyCode::Char('j') => move_selection(app, 1),
        KeyCode::Enter | KeyCode::Char('g') => app.generate(),
        KeyCode::Char(c @ '1'..='9') => app.select_group(c as usize - '1' as usize),
        KeyCode::Char('v') => app.toggle_grid(),
        KeyCode::Char('/') => app.filtering = true,
        KeyCode::Tab => {
            app.focus = Focus::Form;
            app.field_idx = 0;
        }
        KeyCode::Char('q') => app.should_quit = true,
        _ => {}
    }
}

fn handle_form(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Up => app.field_idx = app.field_idx.saturating_sub(1),
        KeyCode::Down => {
            if app.field_idx + 1 < app.fields.len() {
                app.field_idx += 1;
            }
        }
        KeyCode::Enter => app.generate(),
        KeyCode::Tab | KeyCode::Esc => app.focus = Focus::List,
        KeyCode::Backspace => {
            if let Some(field) = app.fields.get_mut(app.field_idx) {
                field.buffer.pop();
            }
        }
        KeyCode::Char(c) => {
            if let Some(field) = app.fields.get_mut(app.field_idx) {
                field.buffer.push(c);
            }
        }
        _ => {}
    }
}

fn move_selection(app: &mut App, delta: isize) {
    let len = app.visible().len();
    if len == 0 {
        app.list_state.select(None);
        return;
    }
    let current = app.list_state.selected().unwrap_or(0) as isize;
    let next = (current + delta).clamp(0, len as isize - 1) as usize;
    app.list_state.select(Some(next));
    app.rebuild_fields();
}

fn reset_selection(app: &mut App) {
    let len = app.visible().len();
    app.list_state.select(if len == 0 { None } else { Some(0) });
    app.rebuild_fields();
}
