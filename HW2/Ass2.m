% 讀取 wav 檔案
[y,Fs] = audioread('star_noisy.wav',[1,16000])
[y2,Fs2] = audioread('star_noisy.wav',[6000,6256])

figure(1)
time1=(1:length(y))/Fs;	% 時間軸的向量
plot(y);
ylim([-1,1])
title("waveform of star_noisy.wav")
xlabel("sample")
ylabel("amplitude")



p = polyfit([0:256],y2,3)
baseline = polyval(p,[0:256])
figure(2)
subplot(2,2,[1,2])
plot(y2)
hold on
plot(baseline)
title("frame and the thrid-order polynomial fit")
xlabel("sample")
ylabel("amplitude")
legend("frame","fit")

y2 = y2 - baseline
figure(3)
plot(y2)
title("detrended frame")
xlabel("sample")
ylabel("amplitude")

[frameMat,Fs2] = audioread('star_noisy.wav',[6000,6256])

[numSamples, numFrames] = size(frameMat);  % 取得每框取樣點數及框數
frameMat2 = zeros(numSamples, numFrames);  % 初始化輸出矩陣

% 建立並 z-normalize 時間軸 (索引)
n = (0:numSamples-1)';               % 時間索引：0, 1, 2, ..., numSamples-1
n_norm = (n - mean(n)) / std(n);     % z-normalization，避免數值不穩定

for i = 1:numFrames
    % 取出第 i 欄 (第 i 個音框)
    frameData = frameMat(:, i);

    % 多項式擬合
    p2 = polyfit(n_norm, frameData, 3);
    baseline2 = polyval(p2, n_norm);

    % 扣除基線得到零漂移修正後的音框
    frameMat2(:, i) = frameData - baseline2;
end
subplot(2,2,[3,4])
plot(frameMat2)
title("detrended frame")
xlabel("sample")
ylabel("amplitude")

