"""
Pipeline Orchestration

Encapsulates the 8-step execution of the MovieRAG self-built ingest pipeline.
Provides clean state tracking, checkpointing, and error handling.
"""

import shutil
import logging
from pathlib import Path

from .config import PreprocessConfig as Cfg

logger = logging.getLogger(__name__)


class PipelineRunner:
    def __init__(self, movie_id: str, video_path: Path, srt_path: Path = None, force: bool = False):
        self.movie_id = movie_id
        self.video_path = video_path.resolve()
        self.srt_path = Path(srt_path).resolve() if srt_path else None
        
        self.target_video = Cfg.RAW_VIDEOS_DIR / f"{self.movie_id}{self.video_path.suffix}"
        self.meta = None
        self.force = force
        self.current_step = ""
        self.failed_step = ""
        self.completed_steps = []
        self.last_error = ""
        self.last_exception = None

    def run_all(self):
        """Execute the full 8-step pipeline."""
        logger.info(f"\n{'=' * 60}")
        logger.info(f"  🎬 INGESTING NEW VIDEO: {self.video_path.name}")
        logger.info(f"  Movie ID: {self.movie_id}")
        logger.info(f"{'=' * 60}")

        try:
            self._start_step("step_1_copy_video")
            self.step_1_copy_video()
            self._complete_step()
            self._start_step("step_2_fetch_metadata")
            self.step_2_fetch_metadata()
            self._complete_step()
            self._start_step("step_3_auto_annotate")
            self.step_3_auto_annotate()
            self._complete_step()
            self._start_step("step_4_stt")
            self.step_4_stt()
            self._complete_step()
            self._start_step("step_4b_semantic_segmentation")
            self.step_4b_semantic_segmentation()
            self._complete_step()
            self._start_step("step_4c_clip_video")
            self.step_4c_clip_video()
            self._complete_step()
            self._start_step("step_5_extract_keyframes")
            self.step_5_extract_keyframes()
            self._complete_step()
            self._start_step("step_5b_cv_face_extraction")
            self.step_5b_cv_face_extraction()
            self._complete_step()
            self._start_step("step_5c_identity_mapping")
            self.step_5c_identity_mapping()
            self._complete_step()
            self._start_step("step_6a_vlm_vision_extraction")
            self.step_6a_vlm_vision_extraction()
            self._complete_step()
            self._start_step("step_6aa_script_scene_vlm_extraction")
            self.step_6aa_script_scene_vlm_extraction()
            self._complete_step()
            self._start_step("step_6b_fusion_graphing")
            auto_clips = self.step_6b_fusion_graphing()
            self._complete_step()
            self._start_step("step_6c_backfill_script_mapping")
            auto_clips = self.step_6c_backfill_script_mapping(auto_clips)
            self._complete_step()
            self._start_step("step_7_build_chunks")
            self.step_7_build_chunks(auto_clips)
            self._complete_step()
            self._start_step("step_7b_build_script_subscenes")
            self.step_7b_build_script_subscenes()
            self._complete_step()
            self._start_step("step_8_index")
            self.step_8_index()
            self._complete_step()
            self._start_step("step_8a_script_scene_index")
            self.step_8a_script_scene_index()
            self._complete_step()
            self._start_step("step_8b_knowledge_graph")
            self.step_8b_knowledge_graph() # New KG Construction Step
            self._complete_step()
            self._start_step("step_8c_sync_graph")
            self.step_8c_sync_graph()
            self._complete_step()
            
            logger.info(f"\n{'=' * 60}")
            logger.info(f"  ✅ INGEST COMPLETE: {self.movie_id}")
            logger.info(f"{'=' * 60}\n")
            return True
        except Exception as e:
            self.last_exception = e
            self.failed_step = self.current_step or "unknown"
            self.last_error = str(e)
            logger.error(f"\n❌ PIPELINE ABORTED for {self.movie_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _start_step(self, step_name: str) -> None:
        self.current_step = step_name

    def _complete_step(self) -> None:
        if self.current_step and self.current_step not in self.completed_steps:
            self.completed_steps.append(self.current_step)

    def is_rate_limited(self) -> bool:
        lowered = str(self.last_error or "").lower()
        markers = (
            "429",
            "rate limit",
            "too many requests",
            "resource_exhausted",
            "quota",
            "rate_limited",
            "scene_keys_unavailable",
            "no active gemini scene keys are available",
            "all gemini scene keys are disabled",
        )
        return any(marker in lowered for marker in markers)

    @staticmethod
    def _should_abort_on_llm_error(exc: Exception) -> bool:
        lowered = str(exc).lower()
        markers = (
            "429",
            "rate limit",
            "too many requests",
            "resource_exhausted",
            "quota",
            "rate_limited",
        )
        return any(marker in lowered for marker in markers)

    def step_1_copy_video(self):
        """Copy the raw video into the managed raw_videos directory."""
        Cfg.RAW_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        if not self.target_video.exists():
            logger.info(f"\n[1/8] Copying video to {self.target_video}...")
            shutil.copy2(str(self.video_path), str(self.target_video))
        else:
            logger.info(f"\n[1/8] Video already at {self.target_video}")

    def step_2_fetch_metadata(self):
        """Crawl OMDB/IMDB for details or create stub."""
        logger.info(f"\n[2/8] Fetching metadata...")
        meta_file = Cfg.get_meta_dir() / f"{self.movie_id}.json"
        
        if meta_file.exists():
            import json
            logger.info("  Metadata already exists, loading.")
            self.meta = json.loads(meta_file.read_text(encoding="utf-8"))
            return

        try:
            from .extractors.metadata_extractor import MetadataCrawler
            crawler = MetadataCrawler()
            self.meta = crawler.crawl(self.movie_id)
        except Exception as e:
            logger.warning(f"  Metadata crawl failed: {e} (continuing with stub)")
            self.meta = {"imdb_id": self.movie_id, "title": self.movie_id}

    def step_3_auto_annotate(self):
        """Run robust PySceneDetect shot boundary detection."""
        ann_path = Cfg.get_annotation_dir() / f"{self.movie_id}.json"
        
        needs_annotation = True
        if ann_path.exists():
            import json
            try:
                with open(ann_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("raw_shots"):
                    needs_annotation = False
            except Exception:
                pass
                
        if needs_annotation or self.force:
            logger.info(f"\n[3/8] Auto-annotating (PySceneDetect shot detection)...")
            from .shotdetect.annotator import AutoAnnotator
            annotator = AutoAnnotator()
            annotation = annotator.annotate(self.movie_id, self.target_video)
            if not annotation:
                logger.warning("  Auto-annotation returned empty, KeyframeExtractor will fallback to fixed intervals.")
        else:
            logger.info(f"\n[3/8] Annotation with raw_shots already exists: {ann_path}")

    def step_4_stt(self):
        """Provide or extract subtitle SRT using whisper/AssemblyAI fallback."""
        target_srt = Cfg.get_subtitle_dir() / f"{self.movie_id}.srt"
        
        if self.srt_path and self.srt_path.exists():
            Cfg.get_subtitle_dir().mkdir(parents=True, exist_ok=True)
            if self.srt_path != target_srt:
                shutil.copy2(str(self.srt_path), str(target_srt))
            logger.info(f"\n[4/8] Using user-provided subtitles at {target_srt}")
            return
            
        if not target_srt.exists():
            logger.info(f"\n[4/8] Generating subtitles (STT)...")
            try:
                from .extractors.audio_stt_extractor import STTGenerator
                stt = STTGenerator()
                stt_result = stt.generate(self.movie_id, self.target_video)
                if not stt_result:
                    logger.warning("  STT generation failed (continuing without subtitles)")
            except Exception as e:
                logger.warning(f"  STT error: {e} (continuing without subtitles)")
        else:
            logger.info(f"\n[4/8] Subtitles already exist: {target_srt}")

    def step_4b_semantic_segmentation(self):
        """Use LLM + Subtitles to group standard optical shots into Semantic Scenes."""
        logger.info(f"\n[4b/8] Semantic Scene Segmentation (LLM + Heuristics)...")
        ann_path = Cfg.get_annotation_dir() / f"{self.movie_id}.json"
        
        # Check if already segmented
        if ann_path.exists():
            import json
            try:
                with open(ann_path, "r", encoding="utf-8") as f:
                    ann_data = json.load(f)
                if ann_data.get("semantic_segmentation", False) and not self.force:
                    logger.info("  Semantic scenes already generated.")
                    return
            except Exception:
                pass
                
        try:
            from .extractors.semantic_scene_segmenter import SemanticSceneSegmenter
            segmenter = SemanticSceneSegmenter()
            success = segmenter.process_movie(self.movie_id)
            if not success:
                logger.warning("  Semantic Scene segmentation skipped/failed. Using fallback optical scenes.")
        except Exception as e:
            if self._should_abort_on_llm_error(e):
                raise
            logger.warning(f"  Semantic Segmenter error: {e}")

    def step_4c_clip_video(self):
        """Physically slice the movie into scene clips using FFmpeg."""
        logger.info(f"\n[4c/8] Physical Video Clipping (MR.Video)...")
        try:
            from .extractors.clip_extractor import ClipExtractor
            extractor = ClipExtractor()
            success = extractor.extract_clips(self.movie_id, self.target_video, force=self.force)
            if not success:
                logger.warning("  Video clipping skipped/failed.")
        except Exception as e:
            logger.warning(f"  Clip Extractor error: {e}")

    def step_5_extract_keyframes(self):
        """Extract 3 precision keyframes for every detected scene."""
        logger.info(f"\n[5/8] Extracting precision keyframes...")
        from .extractors.keyframe_extractor import KeyframeExtractor
        
        extractor = KeyframeExtractor()
        
        # In step 3, annotation might have failed, so we check existence
        ann_path = Cfg.get_annotation_dir() / f"{self.movie_id}.json"
        
        results = []
        if ann_path.exists():
            logger.info(f"  Found annotation boundaries")
            results.append(extractor.process_movie(self.movie_id, force=self.force)) # Pass self.force
        else:
            logger.info(f"  No annotation → extracting at fixed 5s intervals")
            results.append(extractor.process_movie_fixed_interval(self.movie_id, self.target_video, force=self.force)) # Pass self.force

        keyframe_counts = [
            max(
                int(res.get("keyframes", 0) or 0),
                int(res.get("vector_count", 0) or 0),
                int(res.get("vlm_count", 0) or 0),
            )
            for res in results
        ]

        if self.force or any(count > 0 for count in keyframe_counts):
            logger.info("  Keyframes updated.")
        elif any(res.get("status") == "skipped" for res in results):
            logger.info("  Keyframes already exist.")
        else:
            raise RuntimeError("❌ No keyframes extracted! Aborting Pipeline.")

    def step_5b_cv_face_extraction(self):
        """Use CV models (MTCNN/ResNet/DBSCAN) to deterministically track character identities."""
        logger.info(f"\n[5b/8] CV Face Extraction & Identity Clustering...")
        try:
            from .extractors.cv_face_extractor import CVFaceExtractor
            cv_extractor = CVFaceExtractor()
            cv_extractor.process_movie(self.movie_id, force=self.force)
        except Exception as e:
            logger.error(f"  CV Face Extraction skipped/failed: {e}")

    def step_5c_identity_mapping(self):
        """Map DBSCAN face clusters to TMDB character names via One-Shot VLM mapping."""
        logger.info(f"\n[5c/8] One-Shot VLM Identity Mapping...")
        try:
            from .extractors.identity_vlm_mapper import IdentityVLMMapper
            mapper = IdentityVLMMapper()
            mapper.map_identities(self.movie_id)
        except Exception as e:
            logger.error(f"  Identity Mapping failed: {e}")

    def step_6a_vlm_vision_extraction(self):
        """MapReduce batches of scene keyframes into a single VLM sequence prompt."""
        logger.info(f"\n[6a/8] VLM Temporal Vision Extraction...")
        try:
            from .extractors.vlm_vision_extractor import VLMVisionExtractor
            extractor = VLMVisionExtractor()
            extractor.process_movie(self.movie_id, force=self.force)
        except Exception as e:
            if self._should_abort_on_llm_error(e):
                logger.warning(f"  ⚠️ VLM Vision Extraction skipped (rate limit): {e}")
                logger.warning(f"  Pipeline continues — VLM data will be absent for this movie.")
            else:
                logger.error(f"  VLM Vision Extraction failed: {e}")

    def step_6aa_script_scene_vlm_extraction(self):
        """Extract screenplay-aligned visual descriptions for child script scenes."""
        logger.info(f"\n[6aa/8] Script-scene VLM Extraction...")
        try:
            from .extractors.script_scene_vlm_extractor import ScriptSceneVLMExtractor

            extractor = ScriptSceneVLMExtractor()
            extractor.process_movie(self.movie_id, force=self.force)
        except Exception as e:
            if self._should_abort_on_llm_error(e):
                raise
            logger.error(f"  Script-scene VLM Extraction failed: {e}")

    def step_6b_fusion_graphing(self) -> list:
        """Fuse Anonymous VLM Visuals + Deterministic CV Identities + Subtitles into Graph."""
        logger.info(f"\n[6b/8] Identity/Dialogue/Vision Fusion Graphing...")
        try:
            from .extractors.fusion_llm_grapher import FusionLLMGrapher
            grapher = FusionLLMGrapher()
            clips = grapher.process_movie(self.movie_id, force=self.force)
            if not clips:
                logger.warning("  Fusion graphing produced no clips (continuing)")
            return clips
        except Exception as e:
            if self._should_abort_on_llm_error(e):
                raise
            logger.warning(f"  Fusion graphing error: {e} (continuing without)")
            return []

    def step_6c_backfill_script_mapping(self, auto_clips=None) -> list:
        """Refresh screenplay-aware scene mapping after semantic scenes already exist."""
        logger.info(f"\n[6c/8] Screenplay backfill for semantic scenes...")
        try:
            from .extractors.semantic_scene_segmenter import SemanticSceneSegmenter

            segmenter = SemanticSceneSegmenter()
            segmenter.backfill_script_mapping(self.movie_id)

            graph_path = (
                Cfg.get_scene_graph_dir() / self.movie_id / f"{self.movie_id}_auto_graph.json"
            )
            if graph_path.exists():
                import json

                graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
                return graph_data.get("clips", auto_clips or [])
        except Exception as e:
            logger.warning(f"  Screenplay backfill error: {e}")
        return auto_clips or []

    def step_7_build_chunks(self, auto_clips=None):
        """Consolidate JSON tracking outputs into a layered temporal chunk schema."""
        logger.info(f"\n[7/8] Building temporal chunks...")
        from .indexing.chunk_builder import ChunkBuilder
        
        builder = ChunkBuilder()
        # Feed auto_clips to unified_data mock if present
        unified_data = None
        if auto_clips:
             unified_data = {
                 "movies": {
                     self.movie_id: {
                         "clips": auto_clips
                     }
                 }
             }

        chunks = builder.build_for_movie(self.movie_id, unified_data=unified_data)
        
        if chunks:
            import json
            out = Cfg.get_temporal_chunks_dir() / f"{self.movie_id}_chunks.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info(f"  Saved {len(chunks)} chunks → {out}")
            
            pass
        else:
            logger.warning("  No chunks built")

    def step_7b_build_script_subscenes(self):
        """Build derived screenplay sub-scenes for hybrid retrieval."""
        logger.info(f"\n[7b/8] Building screenplay sub-scenes...")
        try:
            from .indexing.script_subscene_builder import ScriptSubsceneBuilder

            builder = ScriptSubsceneBuilder()
            builder.build_for_movie(self.movie_id)
        except Exception as e:
            logger.error(f"  Script sub-scene build failed: {e}")

    def step_8_index(self):
        """FAISS building and Neo4J Graph Enrichment bridging."""
        logger.info(f"\n[8/8] Indexing into FAISS + enriching graph...")
        from .indexing.faiss_builder import FaissBuilder
        from .indexing.graph_builder import GraphBuilder

        chunks_path = Cfg.get_temporal_chunks_dir() / f"{self.movie_id}_chunks.json"

        faiss_builder = FaissBuilder()
        faiss_builder.build_incremental(self.movie_id)

        graph_builder = GraphBuilder()
        graph_builder.enrich(chunks_path=chunks_path, movie_id=self.movie_id)

    def step_8a_script_scene_index(self):
        """Build dedicated screenplay sub-scene text index."""
        logger.info(f"\n[8a/8] Building script-scene retrieval index...")
        try:
            from movierag.indexing.script_scene_indexer import ScriptSceneIndexer

            indexer = ScriptSceneIndexer(index_dir=str(Cfg.get_index_dir()))
            indexer.build_incremental(self.movie_id)
        except Exception as e:
            logger.error(f"  Script-scene indexing failed: {e}")

    def step_8b_knowledge_graph(self):
        """Build Cross-Modal Knowledge Graph (VLM Visuals + Dialogue)."""
        logger.info(f"\n[8b/8] Building Cross-Modal Knowledge Graph...")
        try:
            from .indexing.kg_builder import KnowledgeGraphBuilder
            kg_builder = KnowledgeGraphBuilder(movie_id=self.movie_id)
            kg_builder.build()
        except Exception as e:
            if self._should_abort_on_llm_error(e):
                raise
            logger.error(f"  ❌ KG Construction failed: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def step_8c_sync_graph(self):
        """Sync the built graph artifacts into Neo4j for runtime GraphRAG."""
        logger.info(f"\n[8c/8] Syncing graph artifacts to Neo4j...")
        try:
            from movierag.indexing.neo4j_graph_store import Neo4jGraphStore

            store = Neo4jGraphStore()
            stats = store.sync_movie(self.movie_id)
            if stats.get("synced"):
                logger.info(
                    "  ✅ Neo4j synced: %s chunk nodes, %s script scenes, %s script sub-scenes, %s KG nodes, %s relationships",
                    stats.get("temporal_chunk_nodes", 0),
                    stats.get("script_scene_nodes", 0),
                    stats.get("script_subscene_nodes", 0),
                    stats.get("kg_nodes", 0),
                    stats.get("relationships", 0),
                )
            else:
                logger.warning(
                    "  Neo4j sync skipped: %s",
                    stats.get("reason", "unknown_reason"),
                )
            store.close()
        except Exception as e:
            logger.error(f"  Neo4j graph sync failed: {e}")
