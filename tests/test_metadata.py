"""metadata.py 단위 테스트."""
import io
import os
import struct
import pytest
from PIL import Image, PngImagePlugin


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def clean_png(tmp_path):
    """AI 신호 없는 일반 PNG."""
    img = Image.new("RGB", (32, 32), color=(100, 150, 200))
    p = tmp_path / "clean.png"
    img.save(str(p))
    return str(p)


@pytest.fixture
def sd_parameters_png(tmp_path):
    """Stable Diffusion 'parameters' PNG text chunk 포함."""
    img = Image.new("RGB", (32, 32), color=(100, 150, 200))
    meta = PngImagePlugin.PngInfo()
    meta.add_text("parameters", "masterpiece, best quality, 1girl, Steps: 20, Sampler: DPM++")
    p = tmp_path / "sd_params.png"
    img.save(str(p), pnginfo=meta)
    return str(p)


@pytest.fixture
def comfyui_workflow_png(tmp_path):
    """ComfyUI workflow PNG text chunk 포함."""
    img = Image.new("RGB", (32, 32))
    meta = PngImagePlugin.PngInfo()
    meta.add_text("workflow", '{"nodes": [], "version": "0.4"}')
    p = tmp_path / "comfyui.png"
    img.save(str(p), pnginfo=meta)
    return str(p)


@pytest.fixture
def software_sd_png(tmp_path):
    """Software 키에 Stable Diffusion 포함 PNG."""
    img = Image.new("RGB", (32, 32))
    meta = PngImagePlugin.PngInfo()
    meta.add_text("Software", "Stable Diffusion web UI")
    p = tmp_path / "sw_sd.png"
    img.save(str(p), pnginfo=meta)
    return str(p)


@pytest.fixture
def exif_dalle_jpeg(tmp_path):
    """EXIF Software 태그에 DALL-E 포함 JPEG (raw EXIF bytes 직접 삽입)."""
    img = Image.new("RGB", (32, 32), color=(200, 100, 50))
    p = tmp_path / "dalle.jpg"
    img.save(str(p), format="JPEG")
    # raw 파일에 DALL-E 문자열을 삽입해 XMP/바이트 스캔으로도 탐지 가능하게
    raw = p.read_bytes()
    p.write_bytes(raw + b"Software\x00DALL-E 3\x00")
    return str(p)


@pytest.fixture
def xmp_ai_png(tmp_path):
    """XMP 패킷에 AI 생성 마커 포함 PNG (raw bytes 삽입)."""
    img = Image.new("RGB", (32, 32))
    p = tmp_path / "xmp_ai.png"
    img.save(str(p))
    # 저장된 PNG에 XMP 패킷을 직접 바이트로 삽입
    raw = p.read_bytes()
    xmp_packet = (
        b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        b'<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        b'<rdf:Description rdf:about="" xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/">'
        b'<Iptc4xmpExt:DigitalSourceType>trainedAlgorithmicMedia</Iptc4xmpExt:DigitalSourceType>'
        b'</rdf:Description></rdf:RDF></x:xmpmeta>'
        b'<?xpacket end="w"?>'
    )
    # PNG 마지막 IEND 청크 앞에 텍스트 삽입이 복잡하므로, 파일 끝에 xmp 바이트를 추가
    # (실제 파싱 테스트가 아닌 바이트 패턴 탐지 테스트)
    p.write_bytes(raw + xmp_packet)
    return str(p)


@pytest.fixture
def corrupt_file(tmp_path):
    """손상된 파일 (PNG 헤더만 있고 나머지는 쓰레기)."""
    p = tmp_path / "corrupt.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\xDE\xAD\xBE\xEF" * 10)
    return str(p)


@pytest.fixture
def c2pa_marker_jpeg(tmp_path):
    """JPEG에 C2PA 흔적 바이트 포함."""
    img = Image.new("RGB", (32, 32))
    p = tmp_path / "c2pa.jpg"
    img.save(str(p), format="JPEG")
    raw = p.read_bytes()
    # C2PA JUMBF/APP11 흔적 삽입 (바이트 시그니처)
    c2pa_marker = b"c2pa" + b"\x00" * 4 + b"jumb"
    p.write_bytes(raw + c2pa_marker)
    return str(p)


# ──────────────────────────────────────────────
# TestInspectMetadata: 기본 기능
# ──────────────────────────────────────────────

class TestInspectMetadata:
    """inspect_metadata 핵심 기능 테스트."""

    def test_clean_image_no_signal(self, clean_png):
        from metadata import inspect_metadata
        result = inspect_metadata(clean_png)
        assert result["has_ai_signal"] is False
        assert result["checked"] is True
        assert isinstance(result["signals"], list)
        assert len(result["signals"]) == 0

    def test_sd_parameters_chunk_detected(self, sd_parameters_png):
        """PNG parameters 청크 → has_ai_signal True."""
        from metadata import inspect_metadata
        result = inspect_metadata(sd_parameters_png)
        assert result["has_ai_signal"] is True
        assert result["checked"] is True
        assert len(result["signals"]) > 0

    def test_comfyui_workflow_detected(self, comfyui_workflow_png):
        """ComfyUI workflow 청크 → has_ai_signal True."""
        from metadata import inspect_metadata
        result = inspect_metadata(comfyui_workflow_png)
        assert result["has_ai_signal"] is True
        assert result["checked"] is True

    def test_software_sd_detected(self, software_sd_png):
        """PNG Software 키에 Stable Diffusion → has_ai_signal True."""
        from metadata import inspect_metadata
        result = inspect_metadata(software_sd_png)
        assert result["has_ai_signal"] is True
        assert result["checked"] is True

    def test_exif_dalle_detected(self, exif_dalle_jpeg):
        """EXIF Software = DALL-E → has_ai_signal True."""
        from metadata import inspect_metadata
        result = inspect_metadata(exif_dalle_jpeg)
        assert result["has_ai_signal"] is True
        assert result["checked"] is True

    def test_xmp_trained_algorithmic_media_detected(self, xmp_ai_png):
        """XMP trainedAlgorithmicMedia 마커 → has_ai_signal True."""
        from metadata import inspect_metadata
        result = inspect_metadata(xmp_ai_png)
        assert result["has_ai_signal"] is True
        assert result["checked"] is True

    def test_corrupt_file_no_exception(self, corrupt_file):
        """손상 파일 → 예외 없이 checked=False 반환."""
        from metadata import inspect_metadata
        result = inspect_metadata(corrupt_file)
        # 예외가 나면 이 줄에 도달 못함
        assert result["checked"] is False
        assert isinstance(result["signals"], list)

    def test_nonexistent_file_no_exception(self):
        """존재하지 않는 파일 → 예외 없이 checked=False 반환."""
        from metadata import inspect_metadata
        result = inspect_metadata("/nonexistent/path/image.jpg")
        assert result["checked"] is False

    def test_c2pa_marker_detected(self, c2pa_marker_jpeg):
        """파일 바이트에 c2pa/jumb 시그니처 → 약한 흔적(signals에 기록), has_ai_signal은 decisive 기준.
        C2PA 단순 존재는 decisive 신호가 아니므로 has_ai_signal은 False,
        단 signals 리스트에 weak 흔적이 기록되어야 한다."""
        from metadata import inspect_metadata
        result = inspect_metadata(c2pa_marker_jpeg)
        assert result["checked"] is True
        # C2PA 시그니처는 약한 흔적 — has_ai_signal=False (decisive 기준)
        assert result["has_ai_signal"] is False
        # 단, signals에는 흔적이 기록됨
        assert any("c2pa" in s.lower() or "jumb" in s.lower() for s in result["signals"])

    def test_result_schema_complete(self, clean_png):
        """반환 스키마에 필수 키가 모두 있어야 한다."""
        from metadata import inspect_metadata
        result = inspect_metadata(clean_png)
        assert "has_ai_signal" in result
        assert "signals" in result
        assert "source" in result
        assert "checked" in result

    def test_source_field_on_metadata_signal(self, sd_parameters_png):
        """메타데이터 신호 발견 시 source 필드가 None이 아니어야 한다."""
        from metadata import inspect_metadata
        result = inspect_metadata(sd_parameters_png)
        assert result["source"] is not None

    def test_source_field_none_on_no_signal(self, clean_png):
        """신호 없는 경우 source는 None."""
        from metadata import inspect_metadata
        result = inspect_metadata(clean_png)
        assert result["source"] is None


class TestInspectMetadataEdgeCases:
    """엣지 케이스."""

    def test_empty_string_path(self):
        """빈 문자열 경로 → 예외 없이 checked=False."""
        from metadata import inspect_metadata
        result = inspect_metadata("")
        assert result["checked"] is False

    def test_directory_path(self, tmp_path):
        """디렉토리 경로 → 예외 없이 checked=False."""
        from metadata import inspect_metadata
        result = inspect_metadata(str(tmp_path))
        assert result["checked"] is False

    def test_text_file_no_signal(self, tmp_path):
        """텍스트 파일 → checked=False (이미지 아님)."""
        from metadata import inspect_metadata
        p = tmp_path / "text.txt"
        p.write_text("hello world")
        result = inspect_metadata(str(p))
        assert result["checked"] is False

    def test_midjourney_exif_detected(self, tmp_path):
        """파일 바이트에 Midjourney 문자열 → 탐지."""
        img = Image.new("RGB", (32, 32))
        p = tmp_path / "mj.jpg"
        img.save(str(p), format="JPEG")
        raw = p.read_bytes()
        p.write_bytes(raw + b"Software\x00Midjourney Bot\x00")

        from metadata import inspect_metadata
        result = inspect_metadata(str(p))
        assert result["has_ai_signal"] is True

    def test_firefly_exif_detected(self, tmp_path):
        """파일 바이트에 Adobe Firefly 문자열 → 탐지."""
        img = Image.new("RGB", (32, 32))
        p = tmp_path / "firefly.jpg"
        img.save(str(p), format="JPEG")
        raw = p.read_bytes()
        p.write_bytes(raw + b"Software\x00Adobe Firefly 2.0\x00")

        from metadata import inspect_metadata
        result = inspect_metadata(str(p))
        assert result["has_ai_signal"] is True

    def test_dream_png_chunk_detected(self, tmp_path):
        """AUTOMATIC1111 Dream 청크 탐지."""
        img = Image.new("RGB", (32, 32))
        meta = PngImagePlugin.PngInfo()
        meta.add_text("Dream", "a beautiful landscape, cfg=7.5, steps=30")
        p = tmp_path / "dream.png"
        img.save(str(p), pnginfo=meta)

        from metadata import inspect_metadata
        result = inspect_metadata(str(p))
        assert result["has_ai_signal"] is True

    def test_xmp_content_credentials_detected(self, tmp_path):
        """XMP Content Credentials 마커 → 약한 흔적(signals에 기록).
        'content credentials'는 C2PA 출처 표시이지 AI 단정 근거가 아니므로
        has_ai_signal=False, 단 signals에 기록됨."""
        img = Image.new("RGB", (32, 32))
        p = tmp_path / "cc.png"
        img.save(str(p))
        raw = p.read_bytes()
        xmp_bytes = b"Content Credentials" + b" verified by CAI"
        p.write_bytes(raw + xmp_bytes)

        from metadata import inspect_metadata
        result = inspect_metadata(str(p))
        assert result["checked"] is True
        # content credentials 단독은 약한 흔적
        assert result["has_ai_signal"] is False
        assert any("content credentials" in s.lower() for s in result["signals"])

    def test_xmp_contentauth_detected(self, tmp_path):
        """XMP contentauth 마커 → 약한 흔적(signals에 기록).
        'contentauth'는 C2PA 프로바이더 URI이지 AI 단정 근거가 아니므로
        has_ai_signal=False, 단 signals에 기록됨."""
        img = Image.new("RGB", (32, 32))
        p = tmp_path / "ca.jpg"
        img.save(str(p), format="JPEG")
        raw = p.read_bytes()
        p.write_bytes(raw + b"contentauth:assertion")

        from metadata import inspect_metadata
        result = inspect_metadata(str(p))
        assert result["checked"] is True
        # contentauth 단독은 약한 흔적
        assert result["has_ai_signal"] is False
        assert any("contentauth" in s.lower() for s in result["signals"])


# ──────────────────────────────────────────────
# TestEnsemble: 앙상블 단위 테스트
# ──────────────────────────────────────────────

# ──────────────────────────────────────────────
# TestMetadataFalsePositiveRegression: 오탐 방지 회귀 테스트 (항목 A)
# ──────────────────────────────────────────────

class TestMetadataFileSizeLimit:
    """파일 크기 제한: 50MB 초과 시 raw 바이트 스캔 생략."""

    def test_oversized_file_skips_raw_scan_no_exception(self, tmp_path, monkeypatch):
        """getsize 50MB 초과 시 raw 스캔 생략, 예외 없이 결과 반환."""
        import os as _os
        img = Image.new("RGB", (32, 32), color=(100, 150, 200))
        p = tmp_path / "big.png"
        img.save(str(p))

        from metadata import MAX_METADATA_SCAN_BYTES
        monkeypatch.setattr(_os.path, "getsize", lambda path: MAX_METADATA_SCAN_BYTES + 1)

        from metadata import inspect_metadata
        result = inspect_metadata(str(p))
        # 예외 없이 결과가 반환되어야 함
        assert isinstance(result, dict)
        assert "checked" in result

    def test_oversized_file_checked_is_true(self, tmp_path, monkeypatch):
        """raw 스캔 생략되어도 Pillow 파싱은 정상 동작 — checked=True."""
        import os as _os
        img = Image.new("RGB", (32, 32), color=(100, 150, 200))
        p = tmp_path / "big2.png"
        img.save(str(p))

        from metadata import MAX_METADATA_SCAN_BYTES
        monkeypatch.setattr(_os.path, "getsize", lambda path: MAX_METADATA_SCAN_BYTES + 1)

        from metadata import inspect_metadata
        result = inspect_metadata(str(p))
        assert result["checked"] is True

    def test_oversized_file_signals_note_skip(self, tmp_path, monkeypatch):
        """raw 스캔 생략 시 signals에 관련 메모가 포함된다."""
        import os as _os
        img = Image.new("RGB", (32, 32))
        p = tmp_path / "big3.png"
        img.save(str(p))

        from metadata import MAX_METADATA_SCAN_BYTES
        monkeypatch.setattr(_os.path, "getsize", lambda path: MAX_METADATA_SCAN_BYTES + 1)

        from metadata import inspect_metadata
        result = inspect_metadata(str(p))
        assert any("skip" in s.lower() or "large" in s.lower() for s in result["signals"])

    def test_normal_size_file_raw_scan_runs(self, tmp_path, monkeypatch):
        """정상 크기 파일은 raw 스캔이 실행된다 (Midjourney 문자열 탐지)."""
        import os as _os
        img = Image.new("RGB", (32, 32))
        p = tmp_path / "mj.jpg"
        img.save(str(p), format="JPEG")
        raw = p.read_bytes()
        p.write_bytes(raw + b"Software\x00Midjourney Bot\x00")

        # 실제 파일 크기 사용 (정상 범위)
        from metadata import inspect_metadata
        result = inspect_metadata(str(p))
        assert result["has_ai_signal"] is True


class TestMetadataFalsePositiveRegression:
    """정상 사진이 AI로 오탐되지 않는지 확인하는 회귀 테스트."""

    def test_iphone_description_not_false_positive(self, tmp_path):
        """Description='Photo taken with iPhone' → has_ai_signal=False (오탐 방지)."""
        img = Image.new("RGB", (32, 32), color=(100, 150, 200))
        meta = PngImagePlugin.PngInfo()
        meta.add_text("Description", "Photo taken with iPhone")
        p = tmp_path / "iphone.png"
        img.save(str(p), pnginfo=meta)

        from metadata import inspect_metadata
        result = inspect_metadata(str(p))
        assert result["has_ai_signal"] is False, (
            f"정상 사진이 오탐됨: signals={result['signals']}"
        )

    def test_comment_family_trip_not_false_positive(self, tmp_path):
        """comment='family trip' → has_ai_signal=False (오탐 방지)."""
        img = Image.new("RGB", (32, 32), color=(100, 150, 200))
        meta = PngImagePlugin.PngInfo()
        meta.add_text("comment", "family trip")
        p = tmp_path / "family.png"
        img.save(str(p), pnginfo=meta)

        from metadata import inspect_metadata
        result = inspect_metadata(str(p))
        assert result["has_ai_signal"] is False, (
            f"일반 comment가 오탐됨: signals={result['signals']}"
        )

    def test_sd_parameters_struct_is_decisive(self, tmp_path):
        """SD 파라미터 구조(Steps:, Sampler:, CFG scale:)는 decisive=True, has_ai_signal=True."""
        img = Image.new("RGB", (32, 32), color=(100, 150, 200))
        meta = PngImagePlugin.PngInfo()
        meta.add_text("parameters", "Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 1234, Model hash: abc123")
        p = tmp_path / "sd_struct.png"
        img.save(str(p), pnginfo=meta)

        from metadata import inspect_metadata
        result = inspect_metadata(str(p))
        assert result["has_ai_signal"] is True
        assert result.get("decisive") is True

    def test_exif_adobe_photoshop_not_false_positive(self, tmp_path):
        """EXIF Software='Adobe Photoshop' → has_ai_signal=False."""
        img = Image.new("RGB", (32, 32), color=(200, 100, 50))
        p = tmp_path / "photoshop.jpg"
        img.save(str(p), format="JPEG")
        raw = p.read_bytes()
        p.write_bytes(raw + b"Software\x00Adobe Photoshop 2024\x00")

        from metadata import inspect_metadata
        result = inspect_metadata(str(p))
        assert result["has_ai_signal"] is False, (
            f"Adobe Photoshop이 오탐됨: signals={result['signals']}"
        )

    def test_exif_dalle_is_decisive(self, tmp_path):
        """EXIF/raw에 'DALL-E' → has_ai_signal=True, decisive=True."""
        img = Image.new("RGB", (32, 32), color=(200, 100, 50))
        p = tmp_path / "dalle2.jpg"
        img.save(str(p), format="JPEG")
        raw = p.read_bytes()
        p.write_bytes(raw + b"Software\x00DALL-E 3\x00")

        from metadata import inspect_metadata
        result = inspect_metadata(str(p))
        assert result["has_ai_signal"] is True
        assert result.get("decisive") is True

    def test_schema_has_decisive_field(self, tmp_path):
        """반환 스키마에 decisive 필드가 있어야 한다."""
        img = Image.new("RGB", (32, 32), color=(100, 150, 200))
        p = tmp_path / "schema_check.png"
        img.save(str(p))

        from metadata import inspect_metadata
        result = inspect_metadata(str(p))
        assert "decisive" in result, "decisive 필드가 스키마에 없음"

    def test_comfyui_workflow_with_class_type_is_decisive(self, tmp_path):
        """ComfyUI workflow JSON(class_type 포함) → decisive=True."""
        import json as _json
        img = Image.new("RGB", (32, 32))
        meta = PngImagePlugin.PngInfo()
        workflow = _json.dumps({"nodes": [{"class_type": "KSampler"}], "version": "0.4"})
        meta.add_text("workflow", workflow)
        p = tmp_path / "comfy_decisive.png"
        img.save(str(p), pnginfo=meta)

        from metadata import inspect_metadata
        result = inspect_metadata(str(p))
        assert result["has_ai_signal"] is True
        assert result.get("decisive") is True

    def test_diffusion_standalone_not_false_positive(self, tmp_path):
        """'diffusion' 단어만 있는 Software 태그는 오탐하지 않는다."""
        img = Image.new("RGB", (32, 32), color=(200, 100, 50))
        p = tmp_path / "diffusion_only.jpg"
        img.save(str(p), format="JPEG")
        raw = p.read_bytes()
        # 'diffusion' 단독 — 카메라 앱 이름 같은 케이스
        p.write_bytes(raw + b"Software\x00LightDiffusion Camera App\x00")

        from metadata import inspect_metadata
        result = inspect_metadata(str(p))
        assert result["has_ai_signal"] is False, (
            f"'diffusion' 단독 단어가 오탐됨: signals={result['signals']}"
        )


# ──────────────────────────────────────────────
# TestEnsembleAverageProbability:
    """앙상블: 모델별 평균 확률 계산."""

    def _make_mock_pipeline(self, ai_prob: float):
        """주어진 AI 확률을 반환하는 mock pipeline 팩토리."""
        def pipeline(task, model, **kwargs):
            def infer(image):
                return [
                    {"label": "artificial", "score": ai_prob},
                    {"label": "human", "score": 1.0 - ai_prob},
                ]
            return infer
        return pipeline

    def test_three_models_average(self, valid_image_path):
        """모델 3개의 평균이 올바르게 계산된다."""
        from detector import analyze_image_ensemble

        pipelines = {
            "model_a": self._make_mock_pipeline(0.9),
            "model_b": self._make_mock_pipeline(0.6),
            "model_c": self._make_mock_pipeline(0.3),
        }
        result = analyze_image_ensemble(
            image_path=valid_image_path,
            model_pipelines=pipelines,
            threshold=0.5,
        )
        # 평균 = (0.9 + 0.6 + 0.3) / 3 = 0.6
        assert result["ai_probability"] == pytest.approx(0.6)
        assert result["error"] is None

    def test_ensemble_verdict_based_on_average(self, valid_image_path):
        """평균 확률로 verdict가 결정된다."""
        from detector import analyze_image_ensemble

        # 평균 0.4 → threshold 0.5 → Real
        pipelines = {
            "model_a": self._make_mock_pipeline(0.6),
            "model_b": self._make_mock_pipeline(0.2),
        }
        result = analyze_image_ensemble(
            image_path=valid_image_path,
            model_pipelines=pipelines,
            threshold=0.5,
        )
        assert result["verdict"] == "Real"
        assert result["ai_probability"] == pytest.approx(0.4)

    def test_partial_model_failure_uses_remaining(self, valid_image_path):
        """일부 모델 실패 시 성공한 모델의 평균으로 진행."""
        from detector import analyze_image_ensemble

        def failing_pipeline(task, model, **kwargs):
            raise RuntimeError("Model load failed")

        pipelines = {
            "model_good": self._make_mock_pipeline(0.8),
            "model_fail": failing_pipeline,
        }
        result = analyze_image_ensemble(
            image_path=valid_image_path,
            model_pipelines=pipelines,
            threshold=0.5,
        )
        # 실패한 모델 제외하고 0.8만 사용
        assert result["ai_probability"] == pytest.approx(0.8)
        assert result["error"] is None
        # models 상세에 실패 기록
        failed = [m for m in result["models"] if m["error"] is not None]
        assert len(failed) == 1

    def test_all_models_fail_returns_error(self, valid_image_path):
        """전부 실패 시 error 결과 반환."""
        from detector import analyze_image_ensemble

        def fail(task, model, **kwargs):
            raise RuntimeError("fail")

        pipelines = {"m1": fail, "m2": fail}
        result = analyze_image_ensemble(
            image_path=valid_image_path,
            model_pipelines=pipelines,
            threshold=0.5,
        )
        assert result["error"] is not None
        assert result["ai_probability"] is None
        assert result["verdict"] is None

    def test_models_field_contains_individual_results(self, valid_image_path):
        """models 필드에 모델별 개별 결과가 포함된다."""
        from detector import analyze_image_ensemble

        pipelines = {
            "model_a": self._make_mock_pipeline(0.7),
            "model_b": self._make_mock_pipeline(0.5),
        }
        result = analyze_image_ensemble(
            image_path=valid_image_path,
            model_pipelines=pipelines,
            threshold=0.5,
        )
        assert "models" in result
        assert len(result["models"]) == 2
        model_ids = {m["model"] for m in result["models"]}
        assert "model_a" in model_ids
        assert "model_b" in model_ids

    def test_single_model_in_ensemble_works(self, valid_image_path):
        """모델이 1개일 때도 앙상블 동작."""
        from detector import analyze_image_ensemble

        pipelines = {"solo": self._make_mock_pipeline(0.65)}
        result = analyze_image_ensemble(
            image_path=valid_image_path,
            model_pipelines=pipelines,
            threshold=0.5,
        )
        assert result["ai_probability"] == pytest.approx(0.65)
        assert result["error"] is None

    @pytest.fixture
    def valid_image_path(self, tmp_path):
        img = Image.new("RGB", (32, 32), color=(100, 100, 100))
        p = tmp_path / "test.png"
        img.save(str(p))
        return str(p)


# ──────────────────────────────────────────────
# TestEnsembleConstants: ENSEMBLE_MODELS 상수
# ──────────────────────────────────────────────

class TestEnsembleConstants:
    def test_ensemble_models_defined(self):
        from detector import ENSEMBLE_MODELS
        assert isinstance(ENSEMBLE_MODELS, list)
        assert len(ENSEMBLE_MODELS) >= 2

    def test_default_model_in_ensemble(self):
        from detector import ENSEMBLE_MODELS, DEFAULT_MODEL
        assert DEFAULT_MODEL in ENSEMBLE_MODELS

    def test_ensemble_models_are_strings(self):
        from detector import ENSEMBLE_MODELS
        assert all(isinstance(m, str) for m in ENSEMBLE_MODELS)
