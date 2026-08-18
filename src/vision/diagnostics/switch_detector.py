from dataclasses import dataclass

@dataclass
class TrackEvent:
    track_id: int
    first_frame: int
    last_frame: int
    first_position: tuple[float, float]
    last_position: tuple[float, float]


class SwitchDiagnostics:
    """
    Detecta candidatos a ID switch comparando tracks que desapareceram
    com tracks que apareceram depois, próximos no espaço e no tempo.

    Isso NÃO corrige nada — é só medição, para decidir prioridades.
    """

    def __init__(self, max_gap_frames: int = 45, max_gap_distance: float = 100.0):
        self.max_gap_frames = max_gap_frames
        self.max_gap_distance = max_gap_distance
        self.tracks: dict[int, TrackEvent] = {}
        self.candidates: list[dict] = []

    def update(self, frame_idx: int, detections: list[dict]):
        """
        detections: lista de {"track_id": int, "center": (x, y)}
        """
        seen_this_frame = set()

        for det in detections:
            tid = det["track_id"]
            seen_this_frame.add(tid)

            if tid not in self.tracks:
                self.tracks[tid] = TrackEvent(
                    track_id=tid,
                    first_frame=frame_idx,
                    last_frame=frame_idx,
                    first_position=det["center"],
                    last_position=det["center"],
                )
            else:
                self.tracks[tid].last_frame = frame_idx
                self.tracks[tid].last_position = det["center"]

        # verificar se alguma detecção deste frame "reconecta" com um
        # track morto recentemente (candidato a switch)
        for det in detections:
            tid = det["track_id"]
            if self.tracks[tid].first_frame != frame_idx:
                continue  # não é um track novo, ignora

            for other_tid, other_track in self.tracks.items():
                if other_tid == tid:
                    continue
                if other_tid in seen_this_frame:
                    continue  # ainda está ativo, não é candidato

                gap = frame_idx - other_track.last_frame
                if gap <= 0 or gap > self.max_gap_frames:
                    continue

                dist = self._distance(det["center"], other_track.last_position)
                if dist <= self.max_gap_distance:
                    self.candidates.append({
                        "died_track": other_tid,
                        "died_frame": other_track.last_frame,
                        "died_position": other_track.last_position,
                        "born_track": tid,
                        "born_frame": frame_idx,
                        "born_position": det["center"],
                        "gap_frames": gap,
                        "distance": round(dist, 1),
                    })

    def summary(self):
        total_tracks = len(self.tracks)
        total_candidates = len(self.candidates)
        return {
            "total_track_ids": total_tracks,
            "candidate_switches": total_candidates,
        }

    @staticmethod
    def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5