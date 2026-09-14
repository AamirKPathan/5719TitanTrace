from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class RobotDetection:
    """
    A robot detected in a single video frame.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    alliance: Optional[str] = None
    confidence: float = 0.0

    @property
    def center(self):
        """
        Center point of the robot bounding box.
        """
        x = (self.x1 + self.x2) / 2
        y = (self.y1 + self.y2) / 2
        return x, y


@dataclass
class RobotTrack:
    """
    Persistent identity for one robot throughout a match.
    """

    track_id: int

    alliance: Optional[str] = None
    team_number: Optional[int] = None

    reference_x: float = 0.0
    reference_y: float = 0.0

    last_x: float = 0.0
    last_y: float = 0.0

    last_timestamp: float = 0.0

    visible: bool = True
    missed_frames: int = 0

    trajectory: list = field(default_factory=list)

    def update(self, detection, timestamp):
        """
        Update this robot's position.
        """

        x, y = detection.center

        self.last_x = x
        self.last_y = y
        self.last_timestamp = timestamp

        if detection.alliance is not None:
            self.alliance = detection.alliance

        self.visible = True
        self.missed_frames = 0

        self.trajectory.append(
            {
                "timestamp": timestamp,
                "x": x,
                "y": y,
            }
        )

    def mark_missing(self):
        """
        Mark the robot as temporarily invisible.
        """

        self.visible = False
        self.missed_frames += 1

    def distance_travelled(self):
        """
        Calculate total pixel distance travelled.
        """

        if len(self.trajectory) < 2:
            return 0.0

        distance = 0.0

        for previous, current in zip(
            self.trajectory,
            self.trajectory[1:]
        ):
            dx = current["x"] - previous["x"]
            dy = current["y"] - previous["y"]

            distance += math.sqrt(
                dx * dx + dy * dy
            )

        return distance


class RobotTracker:
    """
    Maintains persistent robot identities.

    This is the foundation for TitanTrace's
    robot reference-point tracking system.
    """

    def __init__(
        self,
        max_distance=150,
        max_missed_frames=30
    ):
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames

        self.tracks = {}
        self.next_track_id = 1

    def initialize(self, detections, timestamp=0.0):
        """
        Initialize robot identities at the beginning
        of a match.

        Each detected robot receives a permanent
        internal track ID.
        """

        self.tracks.clear()
        self.next_track_id = 1

        for detection in detections:
            x, y = detection.center

            track = RobotTrack(
                track_id=self.next_track_id,
                alliance=detection.alliance,
                reference_x=x,
                reference_y=y,
                last_x=x,
                last_y=y,
                last_timestamp=timestamp,
            )

            track.trajectory.append(
                {
                    "timestamp": timestamp,
                    "x": x,
                    "y": y,
                }
            )

            self.tracks[
                self.next_track_id
            ] = track

            self.next_track_id += 1

    def update(self, detections, timestamp):
        """
        Match new detections to existing robot tracks.
        """

        unmatched_detections = list(
            range(len(detections))
        )

        matched_tracks = set()

        # Find nearest detection for each track.
        for track_id, track in self.tracks.items():

            best_index = None
            best_distance = float("inf")

            for index in unmatched_detections:

                detection = detections[index]

                x, y = detection.center

                dx = x - track.last_x
                dy = y - track.last_y

                distance = math.sqrt(
                    dx * dx + dy * dy
                )

                if distance < best_distance:
                    best_distance = distance
                    best_index = index

            if (
                best_index is not None
                and best_distance <= self.max_distance
            ):
                detection = detections[best_index]

                track.update(
                    detection,
                    timestamp
                )

                matched_tracks.add(track_id)

                unmatched_detections.remove(
                    best_index
                )

        # Tracks that weren't seen this frame.
        for track_id, track in self.tracks.items():

            if track_id not in matched_tracks:
                track.mark_missing()

        # New detections become new tracks.
        for index in unmatched_detections:

            detection = detections[index]

            x, y = detection.center

            track = RobotTrack(
                track_id=self.next_track_id,
                alliance=detection.alliance,
                reference_x=x,
                reference_y=y,
                last_x=x,
                last_y=y,
                last_timestamp=timestamp,
            )

            track.trajectory.append(
                {
                    "timestamp": timestamp,
                    "x": x,
                    "y": y,
                }
            )

            self.tracks[
                self.next_track_id
            ] = track

            self.next_track_id += 1

    def get_active_tracks(self):
        """
        Return robots that haven't been missing
        for too long.
        """

        return [
            track
            for track in self.tracks.values()
            if track.missed_frames
            <= self.max_missed_frames
        ]

    def get_track(self, track_id):
        return self.tracks.get(track_id)

    def get_all_tracks(self):
        return list(self.tracks.values())