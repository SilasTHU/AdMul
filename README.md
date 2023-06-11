# AdMul
This is the official repo for Findings of ACL 2023 paper: Adversarial Multi-task Learning for End-to-end Metaphor Detection

# Train
## On VUA All: 

```
python main_vua.py --task all --cfg ./configs/vua_all.yaml
```

## On VUA Verb:
```
python main_vua.py --task verb --cfg ./configs/vua_verb.yaml
```

## On MOH-X:
```
python main_kf.py --task mohx --cfg ./configs/mohx.yaml
```

## On TroFi:
```
python main_kf.py --task trofi --cfg ./configs/trofi.yaml
```
