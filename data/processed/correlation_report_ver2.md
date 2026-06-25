# Phân tích Correlation & Lựa chọn Feature — data_exo_ver2

**Ngày phân tích:** 2026-06-24 | **Dữ liệu:** 4649 dòng (2008-05-01 -> 2026-05-08)

## 1. Cập nhật WTI
Cột `WTI` đã được thay bằng giá Close từ `historical_wti_oil_prices.csv` (khớp theo ngày, forward-fill 119 ngày lễ Mỹ). Giá trị cũ bị "đứng" ở 94.48 cho ~30 dòng cuối; nay đã cập nhật đúng tới gần đây.

## 2. |Correlation| theo MỨC GIÁ (level) với 4 target
|                      |   MG95 |   MG92 |   DO 0.001% |   DO 0.05% |   AVG |
|:---------------------|-------:|-------:|------------:|-----------:|------:|
| MG97                 |  0.999 |  0.998 |       0.907 |      0.952 | 0.964 |
| KERO                 |  0.953 |  0.951 |       0.928 |      0.989 | 0.955 |
| BRT DTD              |  0.976 |  0.981 |       0.895 |      0.953 | 0.951 |
| Brent_EU_Daily       |  0.975 |  0.98  |       0.891 |      0.95  | 0.949 |
| BRT KH               |  0.972 |  0.977 |       0.881 |      0.948 | 0.944 |
| WTI                  |  0.948 |  0.954 |       0.841 |      0.933 | 0.919 |
| WTI_Daily            |  0.948 |  0.953 |       0.838 |      0.93  | 0.917 |
| FO 180               |  0.944 |  0.95  |       0.866 |      0.908 | 0.917 |
| WTI_Monthly          |  0.943 |  0.949 |       0.834 |      0.925 | 0.913 |
| NAPHTHA              |  0.946 |  0.954 |       0.84  |      0.905 | 0.911 |
| Brent_Global_Monthly |  0.944 |  0.951 |       0.833 |      0.901 | 0.907 |
| USD_Index            |  0.44  |  0.46  |       0.283 |      0.369 | 0.388 |
| GPR                  |  0.144 |  0.133 |       0.234 |      0.193 | 0.176 |

> Hầu hết feature đều >0.9 do cùng xu hướng giá dầu (đa cộng tuyến / spurious trend).

## 3. |Correlation| theo BIẾN ĐỘNG NGÀY (Δ, robust hơn)
|                      |   MG95 |   MG92 |   DO 0.001% |   DO 0.05% |   AVG |
|:---------------------|-------:|-------:|------------:|-----------:|------:|
| MG97                 |  0.962 |  0.944 |       0.624 |      0.665 | 0.799 |
| NAPHTHA              |  0.785 |  0.799 |       0.602 |      0.648 | 0.708 |
| FO 180               |  0.73  |  0.741 |       0.611 |      0.646 | 0.682 |
| KERO                 |  0.609 |  0.613 |       0.666 |      0.685 | 0.643 |
| BRT KH               |  0.41  |  0.42  |       0.319 |      0.345 | 0.374 |
| BRT DTD              |  0.403 |  0.412 |       0.318 |      0.345 | 0.37  |
| Brent_EU_Daily       |  0.296 |  0.301 |       0.202 |      0.234 | 0.258 |
| WTI                  |  0.172 |  0.174 |       0.186 |      0.189 | 0.181 |
| WTI_Daily            |  0.145 |  0.146 |       0.117 |      0.129 | 0.134 |
| USD_Index            |  0.119 |  0.12  |       0.076 |      0.096 | 0.103 |
| WTI_Monthly          |  0.063 |  0.067 |       0.068 |      0.066 | 0.066 |
| Brent_Global_Monthly |  0.05  |  0.055 |       0.037 |      0.037 | 0.045 |
| GPR                  |  0.011 |  0.012 |       0.005 |      0.004 | 0.008 |

## 4. Kết luận — Feature tác động lớn nhất
- **MG95 / MG92 (xăng):** mạnh nhất là **MG97, NAPHTHA, FO 180** (cùng nhóm xăng/sản phẩm nhẹ).
- **DO 0.001% / DO 0.05% (diesel):** mạnh nhất là **KERO**, kế đến NAPHTHA, FO 180.
- **Dầu thô (Brent/WTI):** tác động bậc 2 nhưng là benchmark nền tảng.
- **USD_Index:** tương quan yếu (~0.1–0.4) nhưng độc lập với cụm giá → bổ sung thông tin vĩ mô.
- **GPR, các cột _Monthly:** gần như không có tín hiệu ngày → loại.

## 5. Feature giữ lại trong data_exo_ver2
Giữ 7 feature: **MG97, KERO, NAPHTHA, FO 180, BRT DTD, WTI, USD_Index**

Loại 6 cột dư thừa: `BRT KH`, `Brent_EU_Daily` (trùng BRT DTD), `WTI_Daily`, `WTI_Monthly`, `Brent_Global_Monthly` (trùng/làm mượt WTI-Brent), `GPR` (tín hiệu không đáng kể).

→ **data_exo_ver2.csv**: 12 cột = Ngày + 4 target + 7 feature.
