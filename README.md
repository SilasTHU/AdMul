# AdMul
This is the official repo for Findings of ACL 2023 paper: Adversarial Multi-task Learning for End-to-end Metaphor Detection

## data download: 
https://drive.google.com/file/d/12grrfN1BJOcU6ClH4yJxJSymvaF9DY4j/view?usp=drive_link

## checkpoints download: 
vua all: https://drive.google.com/file/d/1STESg8oYHQt71RAkQ9nwhg6TrzrBcJM5/view?usp=drive_link

vua verb: https://drive.google.com/file/d/1pRXxUOXQhQl9-X4aDGPVvWX6fkJgN4JJ/view?usp=drive_link

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
