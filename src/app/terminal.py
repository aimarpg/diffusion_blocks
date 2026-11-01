"""Terminal presentation and keyboard navigation, independent of model inference."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import os
import shutil
import sys
import textwrap
from threading import Event, Thread
import time
from typing import TextIO

try:
    import readline as _readline
except ImportError:
    _readline = None


class ModelSelectionCancelled(Exception):
    """Signal that Escape closed the model picker without choosing a model."""


class Terminal:
    """Present a compact terminal workspace with an accessible plain-text fallback."""

    def __init__(self, plain: bool = False, stream: TextIO | None = None) -> None:
        self.stream = stream if stream is not None else sys.stdout
        self.color = bool(self.stream.isatty() and not plain and 'NO_COLOR' not in os.environ
                          and os.environ.get('TERM') != 'dumb')
        try:
            '─│╭╮╰╯›·⠋'.encode(self.stream.encoding or 'ascii')
            self.unicode = not plain
        except UnicodeEncodeError:
            self.unicode = False
        self.rule = '─' if self.unicode else '-'
        self.rail = '│' if self.unicode else '|'
        self.pointer = '›' if self.unicode else '>'
        self.separator = ' · ' if self.unicode else ' / '
        self._active_models: list[str] = []
        self._session_detail = ''

    @property
    def width(self) -> int:
        """Follow the current terminal width, including window resizing."""
        return max(20, min(96, shutil.get_terminal_size((88, 24)).columns))

    def _write(self, text: str = '') -> None:
        print(text, file=self.stream, flush=True)

    def _style(self, text: str, code: str) -> str:
        return f'\033[{code}m{text}\033[0m' if self.color else text

    def _lines(self, text: str, width: int) -> list[str]:
        # Model/user text never controls cursor position or terminal styling.
        safe = ''.join(char for char in text.expandtabs() if char.isprintable() or char == '\n')
        return [part for line in safe.split('\n')
                for part in (textwrap.wrap(line, width=max(1, width), break_on_hyphens=False) if line else [''])]

    def _body(self, text: str, indent: str = '  ', muted: bool = False) -> None:
        for line in self._lines(text, self.width - len(indent) - 2):
            value = indent + line
            self._write(self._style(value, '2') if muted else value)

    def _edge(self, title: str = '', bottom: bool = False) -> None:
        left = ('╰' if bottom else '╭') if self.unicode else '+'
        for index, text in enumerate(self._lines(title, self.width - 10)):
            part = f' {text} ' if text else ''
            if index == 0:
                length = max(0, self.width - 6 - len(part))
                self._write(self._style('  ' + left + self.rule + part + self.rule * length, '2'))
            else:
                self._body(part, muted=True)

    def heading(self, title: str, subtitle: str) -> None:
        """Show a compact framed identity card instead of an oversized banner."""
        self._write()
        inner = min(self.width, 76) - 6
        tl, tr, bl, br = ('╭', '╮', '╰', '╯') if self.unicode else ('+', '+', '+', '+')
        self._write(self._style('  ' + tl + self.rule * (inner + 2) + tr, '2'))
        rows = [('DiffusionBlocks', '1;36'), (title, '1'), (subtitle, '2')]
        for text, code in rows:
            for line in self._lines(text, inner - 2):
                padding = ' ' * (inner - len(line))
                self._write(self._style('  ' + self.rail + ' ', '2') + self._style(line, code)
                            + padding + self._style(' ' + self.rail, '2'))
        self._write(self._style('  ' + bl + self.rule * (inner + 2) + br, '2'))
        self._write()

    def notice(self, text: str) -> None:
        """Print a subdued instruction or confirmation."""
        self._body(text, muted=True)

    def error(self, text: str) -> None:
        """Render a readable error, with actionable text rather than a traceback."""
        self._write()
        self._edge('Could not continue')
        self._body(text, indent=f'  {self.rail} ')
        self._edge(bottom=True)
        self._write()

    def _catalogue_lines(self, models: dict, groups: dict, available: set[str], default: str | None,
                         numbered: bool, paths: bool, active: str | None = None) -> tuple[list[str], list[str]]:
        lines, order = [], []
        group_names = list(dict.fromkeys(entry.get('group', 'Models') for entry in models.values()))
        for group in group_names:
            members = [(name, entry) for name, entry in models.items()
                       if entry.get('group', 'Models') == group and (not numbered or name in available)]
            if not members:
                continue
            lines.append('')
            lines.extend(self._style('  ' + line, '2') for line in self._lines(groups.get(group, group), self.width - 4))
            for name, entry in members:
                order.append(name)
                prefix = f'{len(order):>2} ' if numbered else ''
                label = entry.get('label', entry['description'])
                state = 'default' if name == default else ('ready' if name in available else 'missing')
                marker = self.pointer if name == active else ' '
                if self.width >= 76:
                    text = f' {marker} {prefix}{name:<13} {label}'
                    room = self.width - len(state) - 6
                    text = text[:room].ljust(room) + f' {state}'
                    lines.append(self._style(text, '1;7') if name == active else text)
                else:
                    text = f' {marker} {prefix}{name}  [{state}]'
                    lines.extend(self._style(line, '1;7') if name == active else line
                                 for line in self._lines(text, self.width - 2))
                    lines.extend(self._style('      ' + part, '2')
                                 for part in self._lines(label, self.width - 8))
                if paths:
                    lines.extend(self._style('      ' + part, '2')
                                 for part in self._lines(entry['checkpoint'], self.width - 8))
        return lines, order

    def catalogue(self, models: dict, groups: dict, available: set[str], default: str | None = None,
                  numbered: bool = False, paths: bool = False) -> list[str]:
        """Show compact, grouped model choices; adapt to narrow terminals."""
        lines, order = self._catalogue_lines(models, groups, available, default, numbered, paths)
        for line in lines:
            self._write(line)
        self._write()
        return order

    def choose_model(self, models: dict, groups: dict, available: set[str], default: str) -> str:
        """Use arrow navigation on a POSIX TTY, with numbered/name input elsewhere."""
        lines, order = self._catalogue_lines(models, groups, available, default, True, False)
        size = shutil.get_terminal_size((88, 24))
        live = (self.color and os.name == 'posix' and sys.stdin.isatty()
                and len(lines) + 4 < size.lines and self.width >= 48)
        if live:
            return self._choose_live(models, groups, available, default, order)
        self.catalogue(models, groups, available, default, numbered=True)
        self.notice(f'Choose a number or name. Enter selects {default}.')
        while True:
            choice = self.ask('Model').strip()
            if not choice:
                return default
            if choice in available:
                return choice
            if choice.isdigit() and 1 <= int(choice) <= len(order):
                return order[int(choice) - 1]
            self.notice('Choose one of the numbers or names above.')

    def _choose_live(self, models: dict, groups: dict, available: set[str], default: str, order: list[str]) -> str:
        import select
        import termios
        import tty

        descriptor = sys.stdin.fileno()
        original = termios.tcgetattr(descriptor)
        current, typed, drawn = order.index(default), '', 0

        def clear() -> None:
            nonlocal drawn
            if drawn:
                self.stream.write(f'\033[{drawn}F')
                self.stream.write(('\033[2K\r\n') * drawn)
                self.stream.write(f'\033[{drawn}F')
                self.stream.flush()
                drawn = 0

        try:
            tty.setraw(descriptor)
            self.stream.write('\033[?25l')
            while True:
                clear()
                lines, _ = self._catalogue_lines(models, groups, available, default, True, False, order[current])
                arrows = '↑/↓' if self.unicode else 'Up/Down'
                hints = ('', f'  {arrows} move   Enter select   Esc cancel',
                         f'  Number or name: {typed}' if typed else '  You can also type a model number or name.')
                lines.extend(self._style(line[:self.width - 2], '2') for line in hints)
                # Raw mode does not translate newline into carriage return.
                self.stream.write('\r\n'.join(lines) + '\r\n')
                self.stream.flush()
                drawn = len(lines)
                key = os.read(descriptor, 1)
                if not key:
                    raise EOFError
                if key == b'\x1b':
                    while select.select([descriptor], [], [], 0.1)[0]:
                        following = os.read(descriptor, 1)
                        if not following:
                            raise EOFError
                        key += following
                        if key[-1:] in (b'A', b'B', b'Z'):
                            break
                if key == b'\x03':
                    raise KeyboardInterrupt
                if key == b'\x1b':
                    raise ModelSelectionCancelled
                if key == b'\x04':
                    raise EOFError
                if key in (b'\x1b[A', b'\x1bOA', b'\x1b[Z'):
                    current = (current - 1) % len(order)
                    typed = ''
                elif key in (b'\x1b[B', b'\x1bOB', b'\t'):
                    current = (current + 1) % len(order)
                    typed = ''
                elif key in (b'\r', b'\n'):
                    if not typed:
                        return order[current]
                    if typed in available:
                        return typed
                    if typed.isdigit() and 1 <= int(typed) <= len(order):
                        return order[int(typed) - 1]
                    typed = ''
                elif key in (b'\x7f', b'\x08'):
                    typed = typed[:-1]
                elif len(key) == 1 and 32 <= key[0] < 127:
                    typed = (typed + key.decode())[:self.width - 20]
        finally:
            clear()
            termios.tcsetattr(descriptor, termios.TCSADRAIN, original)
            self.stream.write('\033[?25h')
            self.stream.flush()

    def ask(self, label: str) -> str:
        """Read input with familiar line editing and in-memory history when available."""
        if label == 'Story opening':
            self._write()
            self._write(self._style('  ' + self.rule * (self.width - 4), '2'))
            self._body(', '.join(self._active_models) + self.separator + self._session_detail, muted=True)
            self.notice('/models switch model   /help shortcuts   /quit exit')
            text = f'  {self.pointer} '
        else:
            text = f'  {label} {self.pointer} '
        if self.color and _readline is not None:
            text = '\001\033[36m\002' + text + '\001\033[0m\002'
        else:
            text = self._style(text, '36')
        return input(text)

    def session(self, device: str, seed: int, tokens: int, models: list[str]) -> None:
        """Show the session context in a compact, aligned summary."""
        self._active_models = models
        self._session_detail = self.separator.join((device.upper(), 'FP32', f'seed {seed}', f'{tokens} new tokens'))
        self._write()
        self._body('model     ' + ', '.join(models))
        self._body('runtime   ' + self.separator.join((device.upper(), 'FP32')), muted=True)
        self._body('sampling  ' + self.separator.join((f'{tokens} new tokens', f'seed {seed}')), muted=True)
        self._write()

    def help(self) -> None:
        """Display the available interactive shortcuts without changing the session."""
        self._write()
        self._edge('Session shortcuts')
        for text in ('/models   Choose another checkpoint', '/help     Show these shortcuts',
                     '/quit     End the session', 'Up/Down   Recall earlier openings when line editing is available'):
            self._body(text, indent=f'  {self.rail} ')
        self._edge(bottom=True)
        self.notice('Each opening starts a new story. Model selection does not retrain the model.')

    def prompt(self, text: str) -> None:
        """Mark the user's opening as a distinct transcript turn."""
        self._write()
        self._write(self._style(f'  {self.pointer} You', '36'))
        self._body(text, indent='    ')
        self._write()

    def model(self, name: str, label: str, index: int, total: int) -> None:
        """Open a model response panel with its comparison position."""
        counter = f'  [{index}/{total}]' if total > 1 else ''
        self._edge(name + counter)
        self._body(label, indent=f'  {self.rail} ', muted=True)

    @contextmanager
    def status(self, message: str) -> Iterator[None]:
        """Show real elapsed time while loading or generating, without fake progress."""
        if not self.color:
            self._body(message + '...', indent=f'  {self.rail} ', muted=True)
            yield
            return
        done = Event()
        start = time.perf_counter()
        frames = ('⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏') if self.unicode else ('|', '/', '-', '\\')

        def animate() -> None:
            index = 0
            while not done.is_set():
                elapsed = time.perf_counter() - start
                line = f'  {self.rail} {frames[index % len(frames)]} {message}  {elapsed:.1f}s'[:self.width - 1]
                self.stream.write('\r\033[2K' + self._style(line, '2'))
                self.stream.flush()
                index += 1
                done.wait(0.1)

        thread = Thread(target=animate, daemon=True)
        thread.start()
        try:
            yield
        finally:
            done.set()
            thread.join()
            self.stream.write('\r\033[2K')
            self.stream.flush()

    def result(self, row: dict) -> None:
        """Present the generated story as a response, followed by a quiet statistics footer."""
        indent = f'  {self.rail} '
        self._body('', indent=indent)
        self._body('Continuation:', indent=indent, muted=True)
        self._body('', indent=indent)
        self._body(row['continuation'] or '[No text before EOS]', indent=indent)
        self._body('', indent=indent)
        details = (f'{row["seconds"]:.2f}s', f'{row["tokens_per_second"]:.1f} tokens/s',
                   f'{row["generated_tokens"]} tokens computed')
        self._edge(self.separator.join(details), bottom=True)
        self._write()
