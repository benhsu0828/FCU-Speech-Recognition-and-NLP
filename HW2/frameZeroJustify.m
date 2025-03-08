function frameMat2 = frameZeroJustify(frameMat, polyOrder)
% frameZeroJustify Zero justification via polynomial fitting
%   frameMat2 = frameZeroJustify(frameMat, polyOrder)
%
%   說明：
%   - frameMat: 輸入的音框矩陣，每一「欄」(column)代表一個音框
%   - polyOrder: 多項式的階數 (例如 3)
%   - frameMat2: 輸出的音框矩陣，已做「零漂移修正」
%
%   每個音框會個別進行：
%   1. 建立時間軸 (或索引) 並做 z-normalization
%   2. 使用 polyfit / polyval 擬合出基線
%   3. 將基線從原音框訊號中扣除，得到去除漂移後的訊號

    [numSamples, numFrames] = size(frameMat);  % 取得每框取樣點數及框數
    frameMat2 = zeros(numSamples, numFrames);  % 初始化輸出矩陣

    % 建立並 z-normalize 時間軸 (索引)
    n = (0:numSamples-1)';               % 時間索引：0, 1, 2, ..., numSamples-1
    n_norm = (n - mean(n)) / std(n);     % z-normalization，避免數值不穩定

    for i = 1:numFrames
        % 取出第 i 欄 (第 i 個音框)
        frameData = frameMat(:, i);

        % 多項式擬合
        p = polyfit(n_norm, frameData, polyOrder);
        baseline = polyval(p, n_norm);

        % 扣除基線得到零漂移修正後的音框
        frameMat2(:, i) = frameData - baseline;
    end
end
