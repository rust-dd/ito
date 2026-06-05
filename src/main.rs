mod app;
mod processes;
mod registry;

use std::io;
use std::time::Duration;

use crossterm::event;
use crossterm::event::Event;
use ratatui::DefaultTerminal;

use crate::app::state::App;

fn main() -> io::Result<()> {
    let mut terminal = ratatui::init();
    let mut app = App::new();
    let result = run(&mut terminal, &mut app);
    ratatui::restore();
    result
}

fn run(terminal: &mut DefaultTerminal, app: &mut App) -> io::Result<()> {
    while !app.should_quit {
        terminal.draw(|frame| crate::app::ui::draw(frame, app))?;
        if event::poll(Duration::from_millis(250))? {
            if let Event::Key(key) = event::read()? {
                crate::app::event::handle_key(app, key);
            }
        }
    }
    Ok(())
}
