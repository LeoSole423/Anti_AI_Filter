import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from degrade_maze import (apply_elastic_warp, build_safety_masks, estimate_wall_width,
                          add_safe_distractors,
                          apply_perspective_transform, build_perspective_transform,
                          calculate_crop_recall, load_image, parse_background,
                          generate_displacement_field, validate_displacement_jacobian,
                          validate_topology)
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

    no_perspective=tmp_path/"out_no_perspective"
    subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/"degrade_maze.py"),str(src),"--output-dir",str(no_perspective),"--seed","423","--no-perspective"],check=True)
    no_perspective_data=json.loads((no_perspective/"run_config.json").read_text())
    assert no_perspective_data["perspective_requested"] is False
    assert all(info["perspective_used"] == 0 for info in no_perspective_data["attempts"].values())
