## 主页信息编辑
_pages/about.md

## 修改顶部导航
__data/navigation.yml新增、删除项，或调整顺序

## 新增Footer
_layouts/default.html末尾中新增
```bash
<div class="footer_site">
<hr/>
<div class="footer_copyright">
  © 2025 PKU · Weihong Zhang | Powered by the&nbsp
  <a class="page_a"  href="https://github.com/RayeRen/acad-homepage.github.io">
  acad
  </a>
</div>
</div>
```
_sass/_page.scss末尾中新增
```bash
.page__related-title {
  margin-bottom: 10px;
  font-size: $type-size-6;
  text-transform: uppercase;
}

.footer_site{
  padding-top: 12em;
  align-items: center;
}
.page_a {
  color: #224b8d;
  text-decoration: none;
  &:hover {
    text-decoration: underline;}
}
hr {
  width: 100%;
  margin: auto;
}

.footer_copyright{
  font-family: $sans-serif-narrow;
  margin-top: 0.5em;
  margin-bottom: 1em;
  display: flex;
  justify-content: center;
  align-items: center;
}
```
