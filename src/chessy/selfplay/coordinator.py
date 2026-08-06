from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import threading
import numpy as np
from chessy.curriculum.manager import CurriculumManager
from chessy.mcts import Evaluator, MCTS, MCTSConfig
from chessy.selfplay.game import SelfPlayGame, play_game
from chessy.selfplay.temperature import TemperatureSchedule
from chessy.observer import TrainingObserver

@dataclass
class SelfPlayCoordinator:
    run_id: str
    run_seed: int
    generation: int
    actors: int
    evaluator: Evaluator
    curriculum: CurriculumManager
    mcts_config: MCTSConfig
    schedule: TemperatureSchedule
    model_checksum: str
    observer: TrainingObserver | None = None
    def run(self, *, games: int, completed_indexes: set[int] | None = None, stop_requested: threading.Event | None = None) -> tuple[list[SelfPlayGame], list[int]]:
        completed_indexes=completed_indexes or set(); token=stop_requested or threading.Event(); assignments=[i for i in range(games) if i not in completed_indexes]
        def one(index:int):
            actor=index % self.actors; rng=np.random.default_rng(int.from_bytes(__import__("hashlib").sha256(f"{self.run_seed}|start|{self.generation}|{index}".encode()).digest()[:8],"big")); start=self.curriculum.sample(rng)
            config=MCTSConfig(simulations=self.mcts_config.simulations,c_puct=self.mcts_config.c_puct,temperature=0.0,root_noise=True,dirichlet_alpha=self.mcts_config.dirichlet_alpha,dirichlet_epsilon=self.mcts_config.dirichlet_epsilon,max_batch_size=self.mcts_config.max_batch_size,max_batch_wait_ms=self.mcts_config.max_batch_wait_ms,seed=index)
            return play_game(run_id=self.run_id,run_seed=self.run_seed,generation=self.generation,game_index=index,actor_id=actor,start=start,mcts=MCTS(self.evaluator,config),schedule=self.schedule,model_checksum=self.model_checksum,stop_requested=token,observer_update=None if self.observer is None else self.observer.live_update)
        completed=[]; incomplete=[]
        with ThreadPoolExecutor(max_workers=self.actors,thread_name_prefix="chessy-selfplay") as pool:
            futures={pool.submit(one,index):index for index in assignments}
            for future in as_completed(futures):
                item=future.result(); (completed.append(item) if item is not None else incomplete.append(futures[future])); self.observer.archive(item, self.model_checksum) if self.observer is not None and item is not None else None
        return sorted(completed,key=lambda item:item.sealed.game_index), sorted(incomplete)
