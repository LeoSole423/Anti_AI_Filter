import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import degrade_maze as dm
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from degrade_maze import (apply_elastic_warp, build_safety_masks, estimate_wall_width,
                          add_safe_distractors,
                          apply_perspective_transform, build_perspective_transform,
                          calculate_crop_recall, load_image, parse_background,
                          generate_displacement_field, validate_displacement_jacobian,
                          validate_topology, ValidationResult, AttemptResult,
                          VariantResult, MAX_VARIANT_ATTEMPTS, generate_variant,
                          validation_failure_reasons, compare_component_identity,
                          build_topology_reference, validate_seed_correspondence,
                          validate_post_render, generate_pseudo_gap_mask,
                          apply_pseudo_gaps, PseudoGapMetrics)
from skimage.measure import euler_number


def synthetic_maze():
    m = np.zeros((180, 240), np.uint8)
    cv2.rectangle(m, (10, 10), (229, 169), 1, 7)
    cv2.line(m, (45, 10), (45, 100), 1, 7)
    cv2.line(m, (45, 100), (150, 100), 1, 7)
    cv2.line(m, (150, 100), (150, 169), 1, 7)
    cv2.line(m, (190, 10), (190, 65), 1, 7)
    cv2.line(m, (190, 65), (90, 65), 1, 7)
    return m


def passing_validation():
    return ValidationResult(
        jacobian=True, wall_components=True, free_space_components=True,
        euler=True, no_crop=True, wall_core_integrity=True,
        jacobian_min=0.95, original_wall_components=1,
        candidate_wall_components=1, original_free_components=1,
        candidate_free_components=1, original_euler=1, candidate_euler=1,
        wall_core_recall=1.0, crop_recall=1.0)


def failing_validation(**overrides):
    values = dict(
        jacobian=False, wall_components=False, free_space_components=False,
        euler=False, no_crop=False, wall_core_integrity=False,
        jacobian_min=0.2, original_wall_components=1,
        candidate_wall_components=2, original_free_components=1,
        candidate_free_components=2, original_euler=1, candidate_euler=2,
        wall_core_recall=0.5, crop_recall=0.5)
    values.update(overrides)
    return ValidationResult(**values)


def test_seed_determinism_and_dimensions():
    m = synthetic_maze(); rng1=np.random.default_rng(423); rng2=np.random.default_rng(423)
    d1=generate_displacement_field(m.shape,rng1,2.0); d2=generate_displacement_field(m.shape,rng2,2.0)
    assert np.array_equal(d1[0],d2[0]) and np.array_equal(d1[1],d2[1])
    assert apply_elastic_warp(m,d1[0],d1[1],cv2.INTER_NEAREST).shape == m.shape


def test_jacobian_positive():
    m=synthetic_maze(); dx,dy=generate_displacement_field(m.shape,np.random.default_rng(1),0.5)
    ok,jmin,_=validate_displacement_jacobian(dx,dy)
    assert ok and jmin > 0.6


def test_same_seed_produces_same_homography():
    t1=build_perspective_transform((120,160),np.random.default_rng(423),0.01,extra_margin_px=8)
    t2=build_perspective_transform((120,160),np.random.default_rng(423),0.01,extra_margin_px=8)
    assert np.array_equal(t1.H,t2.H)
    assert np.array_equal(t1.src,t2.src)
    assert np.array_equal(t1.dst,t2.dst)


def test_rgb_and_mask_share_geometry():
    mask=np.zeros((120,160),np.uint8); cv2.rectangle(mask,(35,30),(95,85),1,-1)
    rgb=np.full((120,160,3),255,np.uint8); rgb[mask>0]=0
    t=build_perspective_transform(mask.shape,np.random.default_rng(8),0.025,extra_margin_px=12)
    warped_rgb=apply_perspective_transform(rgb,t,cv2.INTER_LINEAR,border_value=(255,255,255))
    warped_mask=apply_perspective_transform(mask,t,cv2.INTER_NEAREST,border_value=0)
    dark=warped_rgb.mean(axis=2)<200
    assert np.count_nonzero(dark & (warped_mask>0)) / max(1,np.count_nonzero(warped_mask)) > .97


def test_mask_padding_is_constant_and_does_not_reflect_edges():
    mask=np.zeros((80,100),np.uint8); mask[:, :4]=1
    t=build_perspective_transform(mask.shape,np.random.default_rng(4),0.12,extra_margin_px=12)
    warped=apply_perspective_transform(mask,t,cv2.INTER_NEAREST,border_value=0)
    components, _ = cv2.connectedComponents(warped,8)
    assert components-1 == 1
    assert int(warped[:, 30:].sum()) == 0


def test_perspective_padding_preserves_canvas_without_crop():
    mask=np.zeros((80,100),np.uint8); cv2.rectangle(mask,(0,0),(99,79),1,3)
    t=build_perspective_transform(mask.shape,np.random.default_rng(11),0.10,extra_margin_px=12)
    warped=apply_perspective_transform(mask,t,cv2.INTER_NEAREST,border_value=0)
    ys,xs=np.where(warped>0)
    assert warped.shape == mask.shape and len(xs)>0
    assert 0 <= xs.min() < xs.max() < mask.shape[1]
    assert 0 <= ys.min() < ys.max() < mask.shape[0]


def test_crop_recall_rejects_window_that_cuts_a_wall():
    full=np.zeros((80,100),np.uint8); full[:,20:24]=1
    recall=calculate_crop_recall(full,(22,0,78,80))
    result=validate_topology(full,full,full,full,jacobian_min=.9,crop_recall=recall)
    assert recall < .995
    assert result.no_crop is False


def test_components_euler_and_core_integrity():
    m=synthetic_maze(); width=estimate_wall_width(m); masks=build_safety_masks(m,width)
    dx,dy=generate_displacement_field(m.shape,np.random.default_rng(2),0.15*width)
    candidate=apply_elastic_warp(m,dx,dy,cv2.INTER_NEAREST)
    core=apply_elastic_warp(masks["wall_core"],dx,dy,cv2.INTER_NEAREST)
    result=validate_topology(m,candidate,masks["wall_core"],core,jacobian_min=0.9)
    assert result.wall_components and result.free_space_components and result.euler
    assert result.wall_core_integrity and result.wall_core_recall >= .985


def test_distractor_safety_and_no_original_overwrite(tmp_path):
    src=tmp_path/"input.png"; Image.fromarray(np.full((50,50,3),255,np.uint8)).save(src)
    out=tmp_path/"output"; out.mkdir()
    h=w=300
    wall=np.zeros((h,w),np.uint8); cv2.rectangle(wall,(20,20),(279,279),1,8)
    masks=build_safety_masks(wall,8.0)
    rgb=np.full((h,w,3),245,np.uint8)
    degraded, added=add_safe_distractors(rgb,masks,8.0,np.random.default_rng(5),10.0,np.zeros((h,w),np.uint8))
    assert not np.any((added>0) & (masks["corridor_core"]>0))
    assert degraded.shape == rgb.shape
    marker=tmp_path/"sentinel.txt"; marker.write_text("keep")
    assert marker.read_text()=="keep"


def pseudo_gap_fixture():
    wall=np.zeros((120,160),np.uint8)
    cv2.rectangle(wall,(18,18),(141,101),1,12)
    cv2.line(wall,(55,18),(55,65),1,12)
    masks=build_safety_masks(wall,12.0)
    rgb=np.full((*wall.shape,3),245,np.uint8); rgb[wall>0]=(30,35,45)
    marker=np.zeros_like(wall); marker[18:30,55:67]=1
    return rgb,masks,marker


def test_pseudo_gaps_lighten_ink_and_protect_core_and_markers():
    rgb,masks,marker=pseudo_gap_fixture()
    before_masks={key:value.copy() for key,value in masks.items()}
    gap=generate_pseudo_gap_mask(masks,12.0,np.random.default_rng(423),.36,marker)
    out,metrics=apply_pseudo_gaps(rgb,gap,masks,marker,12.0,.36)
    assert isinstance(metrics,PseudoGapMetrics)
    assert metrics.affected_pixels > 0
    assert metrics.mean_luma_delta > 0
    assert metrics.core_overlap == 0
    assert metrics.marker_overlap == 0
    assert not np.any((gap>0)&(masks["wall_core"]>0))
    assert not np.any((gap>0)&(marker>0))
    assert np.array_equal(out[masks["wall_core"]>0],rgb[masks["wall_core"]>0])
    for key,value in before_masks.items():
        assert np.array_equal(masks[key],value)


def test_pseudo_gaps_are_deterministic_and_coverage_is_monotonic():
    _,masks,marker=pseudo_gap_fixture()
    gaps=[]
    metrics=[]
    for strength in (.16,.36,.60):
        gap=generate_pseudo_gap_mask(masks,12.0,np.random.default_rng(423),strength,marker)
        _,item=apply_pseudo_gaps(np.full((120,160,3),245,np.uint8),gap,masks,marker,12.0,strength)
        gaps.append(gap); metrics.append(item)
    repeat_gap=generate_pseudo_gap_mask(masks,12.0,np.random.default_rng(423),.36,marker)
    repeat_out,repeat_metrics=apply_pseudo_gaps(np.full((120,160,3),245,np.uint8),repeat_gap,masks,marker,12.0,.36)
    reference_out,_=apply_pseudo_gaps(np.full((120,160,3),245,np.uint8),gaps[1],masks,marker,12.0,.36)
    assert np.array_equal(gaps[1],repeat_gap)
    assert np.array_equal(reference_out,repeat_out)
    assert metrics[1] == repeat_metrics
    assert metrics[0].actual_coverage <= metrics[1].actual_coverage <= metrics[2].actual_coverage
    assert metrics[0].requested_coverage < metrics[1].requested_coverage < metrics[2].requested_coverage


def test_pseudo_gaps_never_darken_when_applied():
    rgb,masks,marker=pseudo_gap_fixture()
    gap=generate_pseudo_gap_mask(masks,12.0,np.random.default_rng(9),.60,marker)
    out,_=apply_pseudo_gaps(rgb,gap,masks,marker,12.0,.60)
    assert np.all(out.astype(np.int16) >= rgb.astype(np.int16))


def test_rgba_is_flattened_over_explicit_background(tmp_path):
    rgba=np.zeros((1,1,4),np.uint8); rgba[0,0]=[0,0,0,0]
    src=tmp_path/"transparent.png"; Image.fromarray(rgba,"RGBA").save(src)
    rgb,alpha=load_image(src)
    assert tuple(rgb[0,0]) == (255,255,255)
    assert int(alpha[0,0]) == 0

    rgba[0,0]=[200,0,0,128]; Image.fromarray(rgba,"RGBA").save(src)
    rgb,_=load_image(src)
    expected=Image.alpha_composite(Image.new("RGBA",(1,1),(255,255,255,255)),Image.fromarray(rgba,"RGBA")).convert("RGB")
    assert tuple(rgb[0,0]) == tuple(np.asarray(expected)[0,0])

    normal=np.array([[[12,34,56]]],np.uint8); normal_src=tmp_path/"normal.png"; Image.fromarray(normal,"RGB").save(normal_src)
    normal_rgb,normal_alpha=load_image(normal_src)
    assert np.array_equal(normal_rgb,normal) and int(normal_alpha[0,0]) == 255
    assert parse_background("#f2f2f2") == (242,242,242)
    rgb,_=load_image(src,(242,242,242))
    expected_custom=Image.alpha_composite(Image.new("RGBA",(1,1),(242,242,242,255)),Image.fromarray(rgba,"RGBA")).convert("RGB")
    assert tuple(rgb[0,0]) == tuple(np.asarray(expected_custom)[0,0])


def test_invalid_background_is_rejected(tmp_path):
    import pytest
    with pytest.raises(Exception, match="formato hexadecimal"):
        parse_background("not-a-color")
    src=Path(__file__).resolve().parents[1]/"maze-10x10-kids-1788448203980.png"
    result=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/"degrade_maze.py"),str(src),"--output-dir",str(tmp_path/"bad"),"--levels","subtle","--background","#12"],capture_output=True,text=True)
    assert result.returncode != 0 and "formato hexadecimal" in result.stderr


def test_pdf_generated_and_outputs(tmp_path):
    src=Path(__file__).resolve().parents[1]/"maze-10x10-kids-1788448203980.png"
    source_before=src.read_bytes()
    out=tmp_path/"out"
    subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/"degrade_maze.py"),str(src),"--output-dir",str(out),"--seed","423","--pdf","--debug"],check=True)
    assert (out/"maze_recommended_A4.pdf").exists()
    assert (out/"maze_recommended_A4.png").exists()
    assert Image.open(out/"maze_recommended_A4.png").size==(2480,3508)
    assert Image.open(out/"maze_03_medium.png").mode == "RGB"
    assert Image.open(out/"maze_03_medium.jpg").mode == "RGB"
    data=json.loads((out/"run_config.json").read_text())
    assert src.read_bytes() == source_before
    assert data["perspective_requested"] is True
    assert all(data["attempts"][level]["perspective_used"] > 0 for level in data["attempts"])
    assert all(data["validations"][level]["crop_recall"] >= .995 for level in data["validations"])
    for level in ("subtle","low","medium","strong","max_readable"):
        assert all(data["validations"][level][k] for k in ("jacobian","wall_components","free_space_components","euler","no_crop","wall_core_integrity"))
    for level in ("subtle", "low", "medium", "strong"):
        pseudo=data["attempts"][level]["pseudo_gaps"]
        assert pseudo["actual_coverage"] > 0
        assert pseudo["mean_luma_delta"] > 0
        assert pseudo["core_overlap"] == 0
        assert pseudo["marker_overlap"] == 0

    no_perspective=tmp_path/"out_no_perspective"
    subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/"degrade_maze.py"),str(src),"--output-dir",str(no_perspective),"--seed","423","--no-perspective"],check=True)
    no_perspective_data=json.loads((no_perspective/"run_config.json").read_text())
    assert no_perspective_data["perspective_requested"] is False
    assert all(info["perspective_used"] == 0 for info in no_perspective_data["attempts"].values())


def test_generate_variant_fails_closed_after_max_attempts(tmp_path):
    wall = synthetic_maze()
    masks = build_safety_masks(wall, estimate_wall_width(wall))
    rgb = np.full((*wall.shape, 3), 255, np.uint8)

    def always_fail(*args, **kwargs):
        return failing_validation()

    result = generate_variant(rgb, wall, masks, np.zeros_like(wall), 7.0,
                              "subtle", 423, debug_dir=tmp_path / "debug",
                              validator=always_fail)
    assert result.status == "FAIL"
    assert result.image is None
    assert len(result.attempts) == MAX_VARIANT_ATTEMPTS
    assert result.failure_reasons == ["wall_components", "free_space_components", "euler", "no_crop", "wall_core_integrity"]
    assert (tmp_path / "debug" / "failed" / "subtle" / "attempts.json").exists()
    assert not list(tmp_path.glob("*.png"))
    assert not list(tmp_path.glob("*.jpg"))


def test_jacobian_failure_never_publishes(monkeypatch, tmp_path):
    wall = synthetic_maze()
    masks = build_safety_masks(wall, estimate_wall_width(wall))
    rgb = np.full((*wall.shape, 3), 255, np.uint8)
    monkeypatch.setattr(dm, "validate_displacement_jacobian",
                        lambda dx, dy: (False, 0.1, np.zeros_like(dx)))
    result = generate_variant(rgb, wall, masks, np.zeros_like(wall), 7.0,
                              "low", 423, debug_dir=tmp_path / "debug")
    assert result.status == "FAIL"
    assert result.image is None
    assert len(result.attempts) == MAX_VARIANT_ATTEMPTS
    assert all(not attempt.validation.jacobian for attempt in result.attempts)
    assert "jacobian" in result.failure_reasons


def test_validation_failure_reasons_are_stable():
    result = failing_validation(jacobian=True, wall_components=True,
                                free_space_components=True, euler=True,
                                no_crop=True)
    assert validation_failure_reasons(result) == ["wall_core_integrity"]


def test_partial_success_publishes_only_passes_and_pdf_uses_pass(tmp_path, monkeypatch):
    src = Path(__file__).resolve().parents[1] / "maze-10x10-kids-1788448203980.png"
    original = np.full((48, 64, 3), 245, np.uint8)
    attempt = AttemptResult(1, 1.0, 1.0, 0.01, 0.95, passing_validation(), [], {"perspective_used": 0.01})

    def fake_generate(*args, **kwargs):
        level = args[5]
        if level == "subtle":
            return VariantResult(level, "PASS", original.copy(), passing_validation(), [attempt], [], {})
        return VariantResult(level, "FAIL", None, failing_validation(), [attempt], ["euler"], None)

    monkeypatch.setattr(dm, "generate_variant", fake_generate)
    out = tmp_path / "partial"
    assert dm.main([str(src), "--output-dir", str(out), "--levels", "subtle,low", "--pdf"]) == 0
    config = json.loads((out / "run_config.json").read_text())
    assert config["partial_success"] is True
    assert config["variants"]["subtle"]["status"] == "PASS"
    assert config["variants"]["low"]["status"] == "FAIL"
    assert (out / "maze_01_subtle.png").exists()
    assert (out / "maze_01_subtle.jpg").exists()
    assert not (out / "maze_02_low.png").exists()
    assert not (out / "maze_02_low.jpg").exists()
    assert (out / "maze_recommended_A4.pdf").exists()


def test_all_fail_returns_nonzero_and_does_not_make_recommendation(tmp_path, monkeypatch):
    src = Path(__file__).resolve().parents[1] / "maze-10x10-kids-1788448203980.png"

    def fake_generate(*args, **kwargs):
        level = args[5]
        attempt = AttemptResult(1, 1.0, 1.0, 0.01, 0.2, failing_validation(), ["jacobian"], {})
        return VariantResult(level, "FAIL", None, attempt.validation, [attempt], ["jacobian"], None)

    monkeypatch.setattr(dm, "generate_variant", fake_generate)
    out = tmp_path / "failed"
    assert dm.main([str(src), "--output-dir", str(out), "--levels", "subtle,low", "--pdf"]) == 1
    config = json.loads((out / "run_config.json").read_text())
    assert config["status"] == "FAIL"
    assert config["partial_success"] is False
    assert config["recommended"] is None
    assert config["pdf_generated"] is False
    assert not (out / "maze_recommended_A4.png").exists()
    assert not (out / "maze_recommended_A4.pdf").exists()
    assert not list(out.glob("maze_*.png"))
    assert not list(out.glob("maze_*.jpg"))


def test_component_identity_detects_compensated_split_and_merge():
    expected = np.zeros((80, 140), np.uint16)
    expected[10:35, 10:40] = 1
    expected[10:35, 55:85] = 2
    expected[45:70, 55:85] = 3
    candidate = np.zeros_like(expected, np.uint8)
    candidate[10:22, 10:40] = 1
    candidate[23:35, 10:40] = 1
    candidate[10:35, 55:85] = 1
    candidate[45:70, 55:85] = 1
    candidate[34:47, 65:75] = 1
    result = compare_component_identity(expected, candidate, np.ones_like(candidate), 4)
    assert result.splits > 0
    assert result.merges > 0
    assert not result.passed


def test_free_identity_detects_corridor_closure():
    expected = np.ones((40, 100), np.uint16)
    candidate = np.ones((40, 100), np.uint8)
    candidate[19:21, :] = 0
    result = compare_component_identity(expected, candidate, np.ones_like(candidate), 4)
    assert result.splits > 0
    assert result.merges == 0
    assert not result.passed


def test_identity_ignores_sub_threshold_aliasing():
    expected = np.zeros((50, 70), np.uint16)
    expected[10:40, 10:60] = 1
    candidate = (expected > 0).astype(np.uint8)
    candidate[2, 2] = 1
    result = compare_component_identity(expected, candidate, np.ones_like(candidate), 4)
    assert result.passed
    assert result.splits == result.merges == result.missing == result.unexpected == 0


def test_seed_collision_and_miss_are_reported():
    expected = np.zeros((50, 100), np.uint16)
    expected[15:35, 10:35] = 1
    expected[15:35, 65:90] = 2
    merged = (expected > 0).astype(np.uint8)
    cv2.line(merged, (35, 25), (65, 25), 1, 3)
    reference = build_topology_reference(merged, merged)
    collision = validate_seed_correspondence(expected, [1, 2], merged, np.ones_like(merged), 4)
    assert collision.collisions > 0
    missing = validate_seed_correspondence(expected, [1, 2], (expected == 1).astype(np.uint8), np.ones_like(merged), 4)
    assert missing.misses > 0
    assert reference.wall_ids


def test_core_geometry_rejects_same_area_displacement():
    expected = np.zeros((80, 100), np.uint8)
    expected[20:40, 20:60] = 1
    shifted = np.zeros_like(expected)
    shifted[20:40, 30:70] = 1
    result = validate_topology(expected, shifted, expected, shifted,
                               jacobian_min=.9, crop_recall=1.0,
                               expected_core=expected)
    assert result.wall_core_recall < .995
    assert result.wall_core_precision < .995
    assert not result.core_geometry


def test_post_render_rejects_corridor_closure():
    wall = np.zeros((60, 120), np.uint8)
    cv2.rectangle(wall, (8, 8), (111, 51), 1, 4)
    reference = build_topology_reference(wall, wall)
    expected_wall = reference.wall_labels
    expected_free = reference.free_labels
    domain = reference.domain_mask
    masks = build_safety_masks(wall, 4.0)
    rendered = np.full((60, 120, 3), 245, np.uint8)
    rendered[wall > 0] = (30, 35, 45)
    # A dark bridge closes the free corridor only in the rendered image.
    rendered[28:32, 8:112] = (30, 35, 45)
    post = validate_post_render(rendered, reference, expected_wall, expected_free,
                                domain, masks, np.zeros_like(wall))
    assert not post.passed
    assert post.free_identity.missing > 0 or post.free_identity.splits > 0
