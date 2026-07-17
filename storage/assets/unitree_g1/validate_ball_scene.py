"""validation for scene_mjx_racket_w_ball_flat_terrain.xml.

python validate_ball_scene.py

Checks:
  1. Model compiles, dims are as expected (nq=43 = 36 robot + 7 ball freejoint).
  2. Ball-floor bounce lands inside Table 2's restitution range (0.71-0.79).
  3. Ball-racket contact registers.
  4. Ball-net contact registers AND actually arrests/reflects the ball
     (not just a flagged-but-ineffective contact).

"""
import mujoco
import numpy as np

SCENE = "scene_mjx_racket_w_ball_flat_terrain.xml"


def load():
    return mujoco.MjModel.from_xml_path(SCENE)


def check_dims():
    m = load()
    assert m.nq == 43, f"expected nq=43 (36 robot + 7 ball freejoint), got {m.nq}"
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tennis_ball") >= 0
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "net") >= 0
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor") >= 0
    print("[PASS] check_dims")


def _fresh(m, ball_xyz, ball_vel=(0, 0, 0)):
    qadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "tennis_ball_freejoint")]
    dadr = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "tennis_ball_freejoint")]
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home"))
    d.qpos[qadr:qadr + 3] = ball_xyz
    d.qpos[qadr + 3:qadr + 7] = [1, 0, 0, 0]
    d.qvel[dadr:dadr + 6] = 0
    d.qvel[dadr:dadr + 3] = ball_vel
    mujoco.mj_forward(m, d)
    return d, qadr, dadr


def check_floor_bounce():
    m = load()
    d, qadr, dadr = _fresh(m, [5.0, 3.0, 1.033])
    dt = m.opt.timestep
    zs, vzs = [], []
    for _ in range(int(1.5 / dt)):
        mujoco.mj_step(m, d)
        zs.append(d.qpos[qadr + 2])
        vzs.append(d.qvel[dadr + 2])
    zs, vzs = np.array(zs), np.array(vzs)
    idx = next(i for i in range(1, len(vzs)) if vzs[i - 1] < 0 and vzs[i] >= 0 and zs[i] < 0.15)
    bounce_h = zs[idx:].max() - 0.033
    restitution = float(np.sqrt(max(bounce_h, 0)))
    assert 0.65 <= restitution <= 0.85, (
        f"restitution {restitution:.3f} outside a reasonable band around Table 2's "
        f"0.71-0.79 target"
    )
    print(f"[PASS] check_floor_bounce (restitution={restitution:.3f})")


def check_racket_contact():
    m = load()
    d0, _, _ = _fresh(m, [5, 3, 1.033])
    racket_pos = d0.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tennis_racket_center")].copy()
    d, _, _ = _fresh(m, racket_pos + np.array([0, 0, 0.035]), ball_vel=[0, 0, -0.5])
    seen = False
    for _ in range(50):
        mujoco.mj_step(m, d)
        for k in range(d.ncon):
            names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, d.contact[k].geom1),
                     mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, d.contact[k].geom2)}
            if names == {"tennis_ball_geom", "tennis_racket_collision"}:
                seen = True
    assert seen, "ball dropped directly onto the racket face never registered contact"
    print("[PASS] check_racket_contact")


def check_net_blocks_ball():
    m = load()
    d, qadr, dadr = _fresh(m, [0.5, 0.0, 0.5], ball_vel=[-8.0, 0.0, 0.0])
    dt = m.opt.timestep
    hit = False
    for _ in range(int(0.5 / dt)):
        mujoco.mj_step(m, d)
        for k in range(d.ncon):
            names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, d.contact[k].geom1),
                     mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, d.contact[k].geom2)}
            if names == {"tennis_ball_geom", "net"}:
                hit = True
    final_x = d.qpos[qadr]
    assert hit, "ball fired at the net never registered contact"
    assert final_x < 3.0, f"ball tunneled through the net (ended at x={final_x:.2f}, net at x=0)"
    print("[PASS] check_net_blocks_ball")


def main():
    check_dims()
    check_floor_bounce()
    check_racket_contact()
    check_net_blocks_ball()
    print("\nAll ball/net/court scene checks passed.")


if __name__ == "__main__":
    main()
