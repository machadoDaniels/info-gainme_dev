"""Data types for experiment results analysis."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GameRun:
    """Representa um único jogo (uma linha do CSV).
    
    Attributes:
        target_id: ID do nó objetivo
        target_label: Nome da cidade objetivo
        run_index: Índice da execução (para múltiplas runs da mesma cidade)
        turns: Número de turnos jogados
        h_start: Entropia inicial
        h_end: Entropia final
        total_info_gain: Ganho total de informação
        avg_info_gain_per_turn: Ganho de informação médio por turno
        win: Se o seeker venceu
        compliance_rate: Taxa de compliance do oracle
        conversation_path: Caminho relativo para conversas salvas
    """
    target_id: str
    target_label: str
    run_index: int
    turns: int
    h_start: float
    h_end: float
    total_info_gain: float
    avg_info_gain_per_turn: float
    win: bool
    compliance_rate: float
    conversation_path: Optional[str] = None


@dataclass
class CityStats:
    """Estatísticas agregadas para uma cidade específica.
    
    Attributes:
        city_id: ID da cidade
        city_label: Nome da cidade
        runs: Lista de jogos dessa cidade
    """
    city_id: str
    city_label: str
    runs: list[GameRun]
    
    @property
    def num_runs(self) -> int:
        """Número total de runs desta cidade."""
        return len(self.runs)
    
    @property
    def mean_info_gain(self) -> float:
        """Ganho de informação médio desta cidade."""
        if not self.runs:
            return 0.0
        return sum(r.total_info_gain for r in self.runs) / self.num_runs
    
    @property
    def var_info_gain(self) -> float:
        """Variância do ganho de informação desta cidade."""
        if not self.runs:
            return 0.0
        mean = self.mean_info_gain
        return sum((r.total_info_gain - mean) ** 2 for r in self.runs) / self.num_runs
    
    @property
    def std_info_gain(self) -> float:
        """Desvio padrão do ganho de informação."""
        return self.var_info_gain ** 0.5
    
    @property
    def win_rate(self) -> float:
        """Taxa de vitória nesta cidade."""
        if not self.runs:
            return 0.0
        return sum(1 for r in self.runs if r.win) / self.num_runs
    
    @property
    def mean_turns(self) -> float:
        """Média de turnos nesta cidade."""
        if not self.runs:
            return 0.0
        return sum(r.turns for r in self.runs) / self.num_runs
    
    @property
    def std_turns(self) -> float:
        """Desvio padrão de turnos."""
        if not self.runs:
            return 0.0
        mean = self.mean_turns
        variance = sum((r.turns - mean) ** 2 for r in self.runs) / self.num_runs
        return variance ** 0.5
    
    @property
    def mean_avg_info_gain_per_turn(self) -> float:
        """Média do ganho de informação médio por turno."""
        if not self.runs:
            return 0.0
        return sum(r.avg_info_gain_per_turn for r in self.runs) / self.num_runs
    
    @property
    def std_avg_info_gain_per_turn(self) -> float:
        """Desvio padrão do ganho de informação médio por turno."""
        if not self.runs:
            return 0.0
        mean = self.mean_avg_info_gain_per_turn
        variance = sum((r.avg_info_gain_per_turn - mean) ** 2 for r in self.runs) / self.num_runs
        return variance ** 0.5


@dataclass
class ExperimentResults:
    """Resultados completos de um experimento.
    
    Attributes:
        experiment_name: Nome do experimento
        seeker_model: Modelo usado no Seeker
        oracle_model: Modelo usado no Oracle
        pruner_model: Modelo usado no Pruner
        observability: Modo de observabilidade
        max_turns: Número máximo de turnos
        cities: Dicionário {city_id: CityStats}
    """
    experiment_name: str
    seeker_model: str
    oracle_model: str
    pruner_model: str
    observability: str
    max_turns: int
    cities: dict[str, CityStats]
    
    @property
    def total_runs(self) -> int:
        """Número total de runs no experimento."""
        return sum(city.num_runs for city in self.cities.values())
    
    @property
    def mean_info_gain(self) -> float:
        """Ganho de informação médio global (média das médias por cidade)."""
        if not self.cities:
            return 0.0
        return sum(city.mean_info_gain for city in self.cities.values()) / len(self.cities)
    
    @property
    def global_win_rate(self) -> float:
        """Taxa de vitória global."""
        if self.total_runs == 0:
            return 0.0
        total_wins = sum(sum(1 for r in city.runs if r.win) for city in self.cities.values())
        return total_wins / self.total_runs
    
    @property
    def mean_turns(self) -> float:
        """Média global de turnos."""
        if self.total_runs == 0:
            return 0.0
        total_turns = sum(sum(r.turns for r in city.runs) for city in self.cities.values())
        return total_turns / self.total_runs
    
    @property
    def mean_compliance(self) -> float:
        """Taxa de compliance média global."""
        if self.total_runs == 0:
            return 0.0
        total_compliance = sum(sum(r.compliance_rate for r in city.runs) for city in self.cities.values())
        return total_compliance / self.total_runs
    
    @property
    def mean_avg_info_gain_per_turn(self) -> float:
        """Ganho de informação médio por turno global."""
        if self.total_runs == 0:
            return 0.0
        total_avg_gi = sum(sum(r.avg_info_gain_per_turn for r in city.runs) for city in self.cities.values())
        return total_avg_gi / self.total_runs
    
    def get_city(self, city_id: str) -> Optional[CityStats]:
        """Retorna estatísticas de uma cidade específica."""
        return self.cities.get(city_id)
    
    def summary_dict(self) -> dict:
        """Retorna um dicionário com resumo completo do experimento."""
        return {
            "experiment_name": self.experiment_name,
            "models": {
                "seeker": self.seeker_model,
                "oracle": self.oracle_model,
                "pruner": self.pruner_model,
            },
            "config": {
                "observability": self.observability,
                "max_turns": self.max_turns,
            },
            "global_metrics": {
                "total_runs": self.total_runs,
                "total_cities": len(self.cities),
                "mean_info_gain": round(self.mean_info_gain, 4),
                "mean_avg_info_gain_per_turn": round(self.mean_avg_info_gain_per_turn, 4),
                "win_rate": round(self.global_win_rate, 4),
                "mean_turns": round(self.mean_turns, 2),
                "mean_compliance": round(self.mean_compliance, 4),
            },
            "by_city": {
                city_id: {
                    "label": city.city_label,
                    "runs": city.num_runs,
                    "mean_info_gain": round(city.mean_info_gain, 4),
                    "std_info_gain": round(city.std_info_gain, 4),
                    "var_info_gain": round(city.var_info_gain, 4),
                    "mean_avg_info_gain_per_turn": round(city.mean_avg_info_gain_per_turn, 4),
                    "std_avg_info_gain_per_turn": round(city.std_avg_info_gain_per_turn, 4),
                    "win_rate": round(city.win_rate, 4),
                    "mean_turns": round(city.mean_turns, 2),
                    "std_turns": round(city.std_turns, 2),
                }
                for city_id, city in sorted(self.cities.items())
            }
        }

