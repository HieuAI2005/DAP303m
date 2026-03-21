import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger("eval_framework")


class MovieRAGEvaluator:
    """
    Deterministic regression evaluator for retrieval quality.

    The evaluator focuses on stage-level checks instead of LLM-as-a-judge:
    - scene grounding
    - screenplay alignment / script retrieval
    - visual retrieval
    - graph retrieval
    """

    def __init__(self, pipeline, llm_client, eval_file_path: str):
        self.pipeline = pipeline
        self.llm_client = llm_client
        self.eval_file_path = Path(eval_file_path)
        self.ground_truth = self._load_ground_truth()
        self.results: List[Dict[str, Any]] = []

    def _create_sample_ground_truth(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "scene_titanic_sinking",
                "query": "Tìm cảnh con tàu Titanic chìm trong phim Titanic (1997).",
                "movie_id": "tt0120338",
                "scene_expectation": {
                    "movie_id": "tt0120338",
                    "headings": [
                        "EXT. A-DECK AFT, PORT SIDE",
                        "EXT. STERN",
                    ],
                    "start_time": "02:32:01",
                    "end_time": "02:35:50",
                    "top_k": 5,
                },
                "visual_expectation": {
                    "movie_id": "tt0120338",
                    "chunk_ids": [
                        "tt0120338_chunk_0076",
                        "tt0120338_chunk_0078",
                        "tt0120338_chunk_0081",
                    ],
                    "top_k": 6,
                },
            },
            {
                "id": "script_sixthsense_basement_evening",
                "query": "basement evening",
                "movie_id": "tt0167404",
                "script_expectation": {
                    "movie_id": "tt0167404",
                    "headings": ["INT. BASEMENT - EVENING"],
                    "location": "INT. BASEMENT",
                    "top_k": 4,
                },
                "scene_expectation": {
                    "movie_id": "tt0167404",
                    "headings": ["INT. BASEMENT - EVENING"],
                    "location": "INT. BASEMENT",
                    "start_time": "00:00:00",
                    "end_time": "00:00:53",
                    "top_k": 4,
                },
            },
            {
                "id": "graph_relationship_malcolm_anna",
                "query": "relationship between Malcolm and Anna",
                "movie_id": "tt0167404",
                "graph_expectation": {
                    "movie_id": "tt0167404",
                    "query": "relationship between Malcolm and Anna",
                    "node_type": "CharacterRelation",
                    "character_names": ["Malcolm Crowe", "Anna Crowe"],
                    "question_type": "character_relationship",
                    "top_k": 3,
                },
            },
            {
                "id": "graph_where_basement_evening",
                "query": "where is the basement evening scene",
                "movie_id": "tt0167404",
                "graph_expectation": {
                    "movie_id": "tt0167404",
                    "query": "where is the basement evening scene",
                    "node_type": "ScriptScene",
                    "location": "INT. BASEMENT",
                    "headings": ["INT. BASEMENT - EVENING"],
                    "question_type": "scene_location",
                    "top_k": 4,
                },
            },
        ]

    def _normalize_case(self, raw_case: Dict[str, Any], index: int) -> Dict[str, Any]:
        case = dict(raw_case)
        case.setdefault("id", f"case_{index:03d}")
        case.setdefault("query", "")
        case.setdefault("movie_id", case.get("expected_movie_id", ""))

        # Backward compatibility with the old visual-only dataset.
        if "visual_expectation" not in case and (
            case.get("expected_movie_id") or case.get("expected_shot_ids")
        ):
            case["visual_expectation"] = {
                "movie_id": case.get("expected_movie_id", ""),
                "shot_ids": case.get("expected_shot_ids", []) or [],
                "top_k": 5,
            }

        return case

    def _load_ground_truth(self) -> List[Dict[str, Any]]:
        if not self.eval_file_path.exists():
            logger.warning(
                "Không tìm thấy file eval dataset tại %s. Đang tạo file regression mẫu...",
                self.eval_file_path,
            )
            sample_data = self._create_sample_ground_truth()
            self.eval_file_path.parent.mkdir(parents=True, exist_ok=True)
            self.eval_file_path.write_text(
                json.dumps(sample_data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            return sample_data

        try:
            raw_cases = json.loads(self.eval_file_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("Lỗi đọc file eval: %s", exc)
            return []

        return [
            self._normalize_case(case, index)
            for index, case in enumerate(raw_cases)
            if isinstance(case, dict)
        ]

    @staticmethod
    def _normalize_text(value: Any) -> str:
        return " ".join(str(value or "").lower().split())

    @staticmethod
    def _parse_hms(value: str) -> Optional[int]:
        text = str(value or "").strip()
        if not text:
            return None
        parts = text.split(":")
        if len(parts) != 3:
            return None
        try:
            hours, minutes, seconds = (int(part) for part in parts)
        except Exception:
            return None
        return hours * 3600 + minutes * 60 + seconds

    def _text_matches_any(self, candidate: str, expected_values: List[str]) -> bool:
        candidate_norm = self._normalize_text(candidate)
        if not candidate_norm or not expected_values:
            return False
        for expected in expected_values:
            expected_norm = self._normalize_text(expected)
            if expected_norm and expected_norm in candidate_norm:
                return True
        return False

    def _window_matches(
        self, candidate_start: str, candidate_end: str, expected_start: str, expected_end: str
    ) -> bool:
        if not expected_start and not expected_end:
            return True

        candidate_start_sec = self._parse_hms(candidate_start)
        candidate_end_sec = self._parse_hms(candidate_end)
        expected_start_sec = self._parse_hms(expected_start)
        expected_end_sec = self._parse_hms(expected_end)

        if candidate_start_sec is None or candidate_end_sec is None:
            return False
        if expected_start_sec is None and expected_end_sec is None:
            return True

        expected_start_sec = candidate_start_sec if expected_start_sec is None else expected_start_sec
        expected_end_sec = candidate_end_sec if expected_end_sec is None else expected_end_sec
        return min(candidate_end_sec, expected_end_sec) >= max(
            candidate_start_sec, expected_start_sec
        )

    @staticmethod
    def _result_metadata(result: Any) -> Dict[str, Any]:
        metadata = getattr(result, "metadata", {}) or {}
        return metadata if isinstance(metadata, dict) else {}

    def _summarize_visual_result(self, result: Any) -> Dict[str, Any]:
        metadata = self._result_metadata(result)
        return {
            "movie_id": getattr(result, "movie_id", metadata.get("movie_id", "")),
            "shot_id": metadata.get("shot_id", ""),
            "chunk_id": metadata.get("chunk_id", ""),
            "heading": metadata.get("script_primary_heading", "")
            or metadata.get("script_heading", ""),
            "location": metadata.get("script_location", ""),
            "start_time": metadata.get("start_time", ""),
            "end_time": metadata.get("end_time", ""),
            "score": float(getattr(result, "score", 0.0) or 0.0),
        }

    def _summarize_script_result(self, result: Any) -> Dict[str, Any]:
        metadata = self._result_metadata(result)
        return {
            "movie_id": getattr(result, "movie_id", metadata.get("movie_id", "")),
            "script_scene_uid": metadata.get("script_scene_uid", ""),
            "heading": metadata.get("script_heading", ""),
            "location": metadata.get("script_location", ""),
            "start_time": metadata.get("start_time", ""),
            "end_time": metadata.get("end_time", ""),
            "score": float(getattr(result, "score", 0.0) or 0.0),
        }

    @staticmethod
    def _summarize_scene_result(scene: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "movie_id": scene.get("movie_id", ""),
            "heading": scene.get("heading", ""),
            "location": scene.get("location", ""),
            "start_time": scene.get("start_time", ""),
            "end_time": scene.get("end_time", ""),
            "chunk_ids": scene.get("chunk_ids", []) or [],
            "script_scene_uids": scene.get("script_scene_uids", []) or [],
            "score": float(scene.get("score", 0.0) or 0.0),
        }

    @staticmethod
    def _summarize_graph_hit(hit: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "movie_id": hit.get("movie_id", ""),
            "node_id": hit.get("node_id", ""),
            "node_type": hit.get("node_type", ""),
            "title": hit.get("title", ""),
            "heading": hit.get("heading", ""),
            "location": hit.get("location", ""),
            "character_names": hit.get("character_names", []) or [],
            "question_type": hit.get("question_type", ""),
            "score": float(hit.get("score", 0.0) or 0.0),
        }

    def _evaluate_candidate(self, candidate: Dict[str, Any], expected: Dict[str, Any]) -> tuple[bool, Dict[str, bool]]:
        checks: Dict[str, bool] = {}

        if expected.get("movie_id"):
            checks["movie_id"] = candidate.get("movie_id", "") == expected["movie_id"]
        if expected.get("shot_ids"):
            checks["shot_ids"] = any(
                shot_id in str(candidate.get("shot_id", "")) for shot_id in expected["shot_ids"]
            )
        if expected.get("chunk_ids"):
            candidate_chunk_ids = set(candidate.get("chunk_ids", []) or [])
            if candidate.get("chunk_id"):
                candidate_chunk_ids.add(candidate["chunk_id"])
            checks["chunk_ids"] = bool(candidate_chunk_ids & set(expected["chunk_ids"]))
        if expected.get("script_scene_uids"):
            candidate_uids = set(candidate.get("script_scene_uids", []) or [])
            if candidate.get("script_scene_uid"):
                candidate_uids.add(candidate["script_scene_uid"])
            checks["script_scene_uids"] = bool(candidate_uids & set(expected["script_scene_uids"]))
        if expected.get("headings"):
            checks["headings"] = self._text_matches_any(
                " ".join(
                    str(value or "")
                    for value in (
                        candidate.get("heading", ""),
                        candidate.get("title", ""),
                    )
                ),
                expected["headings"],
            )
        if expected.get("location"):
            checks["location"] = self._text_matches_any(
                candidate.get("location", ""), [expected["location"]]
            )
        if expected.get("node_type"):
            checks["node_type"] = candidate.get("node_type", "") == expected["node_type"]
        if expected.get("question_type"):
            checks["question_type"] = (
                candidate.get("question_type", "") == expected["question_type"]
            )
        if expected.get("character_names"):
            candidate_names = {
                self._normalize_text(value)
                for value in (candidate.get("character_names", []) or [])
                if self._normalize_text(value)
            }
            expected_names = {
                self._normalize_text(value)
                for value in (expected.get("character_names", []) or [])
                if self._normalize_text(value)
            }
            checks["character_names"] = expected_names <= candidate_names
        if expected.get("start_time") or expected.get("end_time"):
            checks["time_window"] = self._window_matches(
                candidate.get("start_time", ""),
                candidate.get("end_time", ""),
                expected.get("start_time", ""),
                expected.get("end_time", ""),
            )

        return all(checks.values()) if checks else False, checks

    def _evaluate_stage(
        self, stage_name: str, candidates: List[Dict[str, Any]], expected: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not expected:
            return {"stage": stage_name, "enabled": False, "passed": None}

        top_k = int(expected.get("top_k", len(candidates) or 1))
        considered = candidates[:top_k]
        best_details = None
        matched_index = None

        for index, candidate in enumerate(considered):
            passed, checks = self._evaluate_candidate(candidate, expected)
            if passed:
                matched_index = index
                best_details = {"candidate": candidate, "checks": checks}
                break
            if best_details is None:
                best_details = {"candidate": candidate, "checks": checks}

        return {
            "stage": stage_name,
            "enabled": True,
            "passed": matched_index is not None,
            "matched_rank": matched_index + 1 if matched_index is not None else None,
            "expected": expected,
            "top_candidates": considered[: min(len(considered), 3)],
            "best_attempt": best_details,
        }

    def run_eval(self):
        total_cases = len(self.ground_truth)
        logger.info("Bắt đầu chạy regression evaluation cho %s cases...", total_cases)

        stage_counters = {"scene": 0, "script": 0, "visual": 0, "graph": 0}
        stage_pass = {"scene": 0, "script": 0, "visual": 0, "graph": 0}
        overall_pass_count = 0

        for index, case in enumerate(self.ground_truth, start=1):
            query = case.get("query", "")
            movie_id = case.get("movie_id") or None
            logger.info("Đang đánh giá case [%s/%s]: %s", index, total_cases, query)
            started_at = time.time()

            visual_results = self.pipeline.retrieve_visual(query, k=6, movie_id=movie_id)
            script_results = self.pipeline.retrieve_script_scenes(query, k=4, movie_id=movie_id)
            scene_results = self.pipeline._build_scene_results(visual_results, script_results, limit=6)

            graph_expectation = case.get("graph_expectation") or {}
            graph_query = graph_expectation.get("query", query)
            graph_hits = self.pipeline.query_graph(
                graph_query,
                movie_id=graph_expectation.get("movie_id") or movie_id,
                limit=int(graph_expectation.get("top_k", 4)),
            ) if graph_expectation else []

            scene_eval = self._evaluate_stage(
                "scene",
                [self._summarize_scene_result(scene) for scene in scene_results],
                case.get("scene_expectation"),
            )
            script_eval = self._evaluate_stage(
                "script",
                [self._summarize_script_result(result) for result in script_results],
                case.get("script_expectation"),
            )
            visual_eval = self._evaluate_stage(
                "visual",
                [self._summarize_visual_result(result) for result in visual_results],
                case.get("visual_expectation"),
            )
            graph_eval = self._evaluate_stage(
                "graph",
                [self._summarize_graph_hit(hit) for hit in graph_hits],
                case.get("graph_expectation"),
            )

            enabled_stage_results = [
                stage_result
                for stage_result in (scene_eval, script_eval, visual_eval, graph_eval)
                if stage_result.get("enabled")
            ]
            for stage_result in enabled_stage_results:
                stage_name = stage_result["stage"]
                stage_counters[stage_name] += 1
                if stage_result.get("passed"):
                    stage_pass[stage_name] += 1

            case_passed = bool(enabled_stage_results) and all(
                stage_result.get("passed") for stage_result in enabled_stage_results
            )
            if case_passed:
                overall_pass_count += 1

            elapsed_time = round(time.time() - started_at, 2)
            result_record = {
                "id": case.get("id"),
                "query": query,
                "movie_id": movie_id,
                "latency_sec": elapsed_time,
                "passed": case_passed,
                "scene": scene_eval,
                "script": script_eval,
                "visual": visual_eval,
                "graph": graph_eval,
            }
            self.results.append(result_record)
            logger.info(
                "Case %s | pass=%s | scene=%s | script=%s | visual=%s | graph=%s | latency=%.2fs",
                case.get("id"),
                case_passed,
                scene_eval.get("passed"),
                script_eval.get("passed"),
                visual_eval.get("passed"),
                graph_eval.get("passed"),
                elapsed_time,
            )

        summary = {
            "total_cases": total_cases,
            "overall_pass_rate": overall_pass_count / total_cases if total_cases else 0.0,
        }
        for stage_name in ("scene", "script", "visual", "graph"):
            count = stage_counters[stage_name]
            summary[f"{stage_name}_pass_rate"] = stage_pass[stage_name] / count if count else None
            summary[f"{stage_name}_cases"] = count

        report = {
            "summary": summary,
            "details": self.results,
        }

        report_path = self.eval_file_path.parent / "eval_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("✅ Regression evaluation hoàn tất. Báo cáo lưu tại: %s", report_path)
        logger.info(
            "Tổng quan: overall=%.1f%% | scene=%s | script=%s | visual=%s | graph=%s",
            summary["overall_pass_rate"] * 100.0,
            f"{summary['scene_pass_rate'] * 100.0:.1f}%" if summary["scene_pass_rate"] is not None else "n/a",
            f"{summary['script_pass_rate'] * 100.0:.1f}%" if summary["script_pass_rate"] is not None else "n/a",
            f"{summary['visual_pass_rate'] * 100.0:.1f}%" if summary["visual_pass_rate"] is not None else "n/a",
            f"{summary['graph_pass_rate'] * 100.0:.1f}%" if summary["graph_pass_rate"] is not None else "n/a",
        )
        return report
