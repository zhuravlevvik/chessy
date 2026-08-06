"""Correct, single-threaded ``mcts-puct-v1`` tree search."""

from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np

from chessy.chess import ChessEnvironment
from chessy.encoding import ACTION_SIZE, decode_action, encode_move
from chessy.mcts.config import MCTSConfig
from chessy.mcts.evaluator import Evaluation, Evaluator
from chessy.mcts.node import SearchNode, backup, select_child


@dataclass(frozen=True, slots=True)
class SearchAction:
    visits: int
    probability: float
    prior: float


@dataclass(frozen=True, slots=True)
class SearchResult:
    action: int
    move: chess.Move
    policy: dict[int, SearchAction]
    root_value: float
    simulations: int


def visit_policy(result: SearchResult) -> dict[int, float]:
    """Return the raw-root-visit target, independent of move temperature."""
    if not result.policy:
        raise ValueError("cannot build a policy from an empty root")
    actions = sorted(result.policy)
    visits = np.asarray([result.policy[action].visits for action in actions], dtype=np.float64)
    if not np.isfinite(visits).all() or (visits < 0).any():
        raise ValueError("root visits must be finite and non-negative")
    total = float(visits.sum())
    if total <= 0:
        raise ValueError("root visit total must be positive")
    return {action: float(count / total) for action, count in zip(actions, visits, strict=True)}


def sample_action(result: SearchResult, *, temperature: float, rng: np.random.Generator) -> int:
    """Sample a move from raw visits without modifying the saved MCTS target."""
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or not np.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be a finite non-negative number")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy Generator")
    target = visit_policy(result)
    actions = np.asarray(sorted(target), dtype=np.int64)
    visits = np.asarray([result.policy[int(action)].visits for action in actions], dtype=np.float64)
    if temperature == 0:
        best = int(visits.max())
        return min(int(action) for action, count in zip(actions, visits, strict=True) if int(count) == best)
    weights = np.power(visits, 1.0 / float(temperature))
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("temperature sampling requires positive visit weights")
    return int(rng.choice(actions, p=weights / total))


def position_fingerprint(environment: ChessEnvironment) -> tuple[str, ...]:
    # All current-to-past states that affect board119-v1, including repetition state.
    return tuple(
        f"{board.fen(en_passant='fen')}|r2={int(board.is_repetition(2))}|r3={int(board.is_repetition(3))}"
        for board in environment.history()
    )


class MCTS:
    def __init__(self, evaluator: Evaluator, config: MCTSConfig = MCTSConfig()) -> None:
        self.evaluator = evaluator
        self.config = config
        self._root: SearchNode | None = None
        self._rng = np.random.default_rng(config.seed)

    @property
    def root(self) -> SearchNode | None:
        return self._root

    def reset(self) -> None:
        self._root = None

    def _expand(self, node: SearchNode, environment: ChessEnvironment, evaluation: Evaluation) -> None:
        board = environment.board
        policy = np.asarray(evaluation.policy, dtype=np.float64)
        if policy.shape != (ACTION_SIZE,) or not np.isfinite(policy).all() or (policy < 0).any():
            raise ValueError(f"evaluator policy must be finite, non-negative, and shape [{ACTION_SIZE}]")
        legal: list[tuple[int, ChessEnvironment]] = []
        seen: set[int] = set()
        for move in board.legal_moves:
            action = encode_move(board, move)
            if action in seen:
                raise ValueError(f"legal action collision at index {action}")
            seen.add(action)
            child_environment = environment.copy()
            child_environment.push(move)
            legal.append((action, child_environment))
        total = float(sum(policy[action] for action, _ in legal))
        if not np.isfinite(total) or total <= 0:
            priors = {action: 1.0 / len(legal) for action, _ in legal}
        else:
            priors = {action: float(policy[action] / total) for action, _ in legal}
        node.children = {
            action: SearchNode(
                prior=priors[action],
                to_play=child_environment.board.turn,
                fingerprint=position_fingerprint(child_environment),
            )
            for action, child_environment in legal
        }
        node.expanded = True

    def _apply_root_noise(self, root: SearchNode) -> None:
        if root.noise_applied or not self.config.root_noise or not root.children:
            return
        actions = sorted(root.children)
        noise = self._rng.dirichlet(np.full(len(actions), self.config.dirichlet_alpha))
        epsilon = self.config.dirichlet_epsilon
        for action, sample in zip(actions, noise, strict=True):
            child = root.children[action]
            child.prior = (1 - epsilon) * child.prior + epsilon * float(sample)
        total = sum(child.prior for child in root.children.values())
        for child in root.children.values():
            child.prior /= total
        root.noise_applied = True

    @staticmethod
    def _terminal_value(environment: ChessEnvironment) -> float:
        return environment.terminal_value(environment.board.turn)

    def _ensure_root(self, environment: ChessEnvironment) -> SearchNode:
        fingerprint = position_fingerprint(environment)
        if self._root is None or self._root.fingerprint != fingerprint:
            self._root = SearchNode(prior=1.0, to_play=environment.board.turn, fingerprint=fingerprint)
        return self._root

    def search(self, environment: ChessEnvironment) -> SearchResult:
        if environment.is_terminal():
            raise ValueError("cannot search a terminal position")
        root = self._ensure_root(environment)
        if not root.expanded:
            evaluation = self.evaluator.evaluate(environment.history())
            self._expand(root, environment, evaluation)
        self._apply_root_noise(root)

        for _ in range(self.config.simulations):
            simulation = environment.copy()
            node = root
            path = [root]
            while node.expanded and node.children:
                action, node = select_child(node, self.config.c_puct)
                simulation.push(decode_action(simulation.board, action))
                path.append(node)
            if simulation.is_terminal():
                value = self._terminal_value(simulation)
            else:
                evaluation = self.evaluator.evaluate(simulation.history())
                value = float(np.clip(evaluation.value, -1.0, 1.0))
                self._expand(node, simulation, evaluation)
            backup(path, value)

        visits = np.array([root.children[action].visit_count for action in sorted(root.children)], dtype=np.float64)
        actions = np.array(sorted(root.children), dtype=np.int64)
        if visits.sum() <= 0:
            raise ValueError("MCTS completed without root visits")
        probabilities = visits / visits.sum()
        visit_policy = {
            int(action): SearchAction(
                visits=int(root.children[int(action)].visit_count),
                probability=float(probability),
                prior=float(root.children[int(action)].prior),
            )
            for action, probability in zip(actions, probabilities, strict=True)
        }
        provisional = SearchResult(
            action=int(actions[0]),
            move=decode_action(environment.board, int(actions[0])),
            policy=visit_policy,
            root_value=root.mean_value,
            simulations=self.config.simulations,
        )
        chosen = sample_action(provisional, temperature=self.config.temperature, rng=self._rng)
        return SearchResult(
            action=chosen, move=decode_action(environment.board, chosen), policy=visit_policy,
            root_value=root.mean_value, simulations=self.config.simulations,
        )

    def advance(self, environment: ChessEnvironment, action: int) -> bool:
        fingerprint = position_fingerprint(environment)
        if self._root is not None and action in self._root.children:
            child = self._root.children[action]
            if child.fingerprint == fingerprint:
                self._root = child
                return True
        self._root = SearchNode(prior=1.0, to_play=environment.board.turn, fingerprint=fingerprint)
        return False
