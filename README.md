# AdMul
This is the official repo for Findings of ACL 2023 paper: Adversarial Multi-task Learning for End-to-end Metaphor Detection

# Train
## On VUA All: 

```
python main_vua.py --task all --cfg ./config/vua_all.yaml
```

## On VUA Verb:
```
python main_vua.py --task verb --cfg ./config/vua_verb.yaml
```

## On MOH-X:
```
python main_kf.py --task mohx --cfg ./config/mohx.yaml
```

## On TroFi:
```
python main_kf.py --task trofi --cfg ./config/trofi.yaml
```
