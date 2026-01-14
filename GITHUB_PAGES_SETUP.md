# GitHub Pages 설정 가이드

이 문서는 GRCWA MkDocs 문서를 GitHub Pages에 배포하는 방법을 설명합니다.

## 🔧 GitHub Repository 설정

### 1. GitHub Pages 활성화

1. GitHub 저장소로 이동
2. **Settings** 탭 클릭
3. 왼쪽 사이드바에서 **Pages** 클릭
4. **Source** 섹션에서:
   - **Source**: `GitHub Actions` 선택 ⚠️ 중요!
   - ~~Deploy from a branch~~ ❌ 이것이 아님

![GitHub Pages Source Setting](https://docs.github.com/assets/cb-47267/mw-1440/images/help/pages/publishing-source-drop-down.webp)

### 2. Actions 권한 확인

1. **Settings** > **Actions** > **General**
2. **Workflow permissions** 섹션에서:
   - ✅ "Read and write permissions" 선택
   - ✅ "Allow GitHub Actions to create and approve pull requests" 체크

## 🚀 배포 방법

### 자동 배포 (권장)

다음 브랜치에 push하면 자동으로 배포됩니다:

```bash
# main 브랜치에 push
git push origin main

# 또는 현재 개발 브랜치
git push origin claude/add-rcwa-documentation-FQA79
```

### 수동 배포

GitHub 웹사이트에서:

1. **Actions** 탭으로 이동
2. 왼쪽에서 "Deploy MkDocs Documentation" 선택
3. **Run workflow** 버튼 클릭
4. 브랜치 선택 후 **Run workflow** 클릭

## 📊 배포 상태 확인

### Actions 탭에서 확인

1. **Actions** 탭 클릭
2. 최근 워크플로우 실행 확인
3. 두 개의 job이 성공해야 함:
   - ✅ **build**: MkDocs 사이트 빌드
   - ✅ **deploy**: GitHub Pages에 배포

### 배포 URL 확인

배포가 성공하면:

- **개인 계정**: `https://[username].github.io/grcwa/`
- **조직**: `https://[organization].github.io/grcwa/`

예시: `https://geniuskey.github.io/grcwa/`

## 🐛 문제 해결

### 404 에러 발생

**원인 1: Source 설정이 잘못됨**

✅ **해결**: Settings > Pages > Source를 `GitHub Actions`로 변경

**원인 2: 배포가 완료되지 않음**

✅ **해결**: Actions 탭에서 워크플로우가 성공했는지 확인

**원인 3: Repository가 Private**

✅ **해결**:
- Settings > General > Danger Zone
- "Change visibility" 클릭
- Public으로 변경 (무료 계정의 경우)

또는 GitHub Pro/Team/Enterprise 계정 사용

**원인 4: Actions 권한 부족**

✅ **해결**: Settings > Actions > General > Workflow permissions에서 "Read and write permissions" 선택

### 배포 실패

**빌드 에러 확인**

```bash
# 로컬에서 테스트
mkdocs build --strict
```

에러가 발생하면 수정 후 다시 push

**필수 플러그인 누락**

Actions 로그에서 import 에러 확인:
- mkdocs-material
- mkdocstrings
- mkdocs-git-revision-date-localized-plugin

### 스타일이 깨짐

**원인**: CSS/JS 파일 경로 문제

✅ **해결**: 모든 경로가 상대 경로인지 확인

```markdown
<!-- ✅ 올바른 경로 -->
![image](../imag/scheme.png)

<!-- ❌ 잘못된 경로 -->
![image](/imag/scheme.png)
```

## 🔄 워크플로우 설명

현재 GitHub Actions 워크플로우 (`.github/workflows/docs.yml`):

```yaml
# 트리거: main, master, 또는 개발 브랜치에 push
on:
  push:
    branches:
      - main
      - master
      - claude/add-rcwa-documentation-FQA79
  workflow_dispatch:  # 수동 실행 가능

# 두 개의 job으로 구성
jobs:
  build:
    # 1. Python 설정
    # 2. 의존성 설치
    # 3. MkDocs 빌드
    # 4. artifact 업로드

  deploy:
    # 1. artifact 다운로드
    # 2. GitHub Pages에 배포
```

## 📝 로컬 개발

배포 전 로컬에서 테스트:

```bash
# 의존성 설치
pip install mkdocs-material
pip install mkdocstrings[python]
pip install pymdown-extensions
pip install mkdocs-git-revision-date-localized-plugin
pip install pillow cairosvg

# 로컬 서버 실행 (http://127.0.0.1:8000)
mkdocs serve

# 빌드 테스트
mkdocs build --clean --strict

# 빌드 결과 확인
ls -la site/
```

## ✅ 체크리스트

배포 전 확인사항:

- [ ] Repository가 Public이거나 GitHub Pro 이상
- [ ] Settings > Pages > Source가 "GitHub Actions"
- [ ] Settings > Actions > Workflow permissions가 "Read and write"
- [ ] `.github/workflows/docs.yml` 파일 존재
- [ ] `mkdocs.yml` 설정이 올바름
- [ ] 로컬에서 `mkdocs build` 성공
- [ ] 모든 링크와 이미지 경로 확인

## 🎯 다음 단계

1. **Settings > Pages** 에서 Source를 "GitHub Actions"로 변경
2. 현재 브랜치에 작은 변경사항 commit & push
3. Actions 탭에서 배포 진행 상황 확인
4. 배포 완료 후 URL 접속 테스트

## 📞 추가 도움이 필요한 경우

- GitHub Pages 공식 문서: https://docs.github.com/en/pages
- MkDocs Material 문서: https://squidfunk.github.io/mkdocs-material/
- Issue 등록: https://github.com/weiliangjinca/grcwa/issues

---

**참고**: 첫 배포는 5-10분 정도 소요될 수 있습니다. 인내심을 가지고 기다려주세요! 🕐
