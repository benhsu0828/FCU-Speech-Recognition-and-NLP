#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<limits.h>
#include<time.h>

#define targetString "intensation"

int min(int a, int b, int c) {
    if (a <= b && a <= c) return a;
    if (b <= a && b <= c) return b;
    return c;
}

typedef struct {
    char origin_word[100];
    char new_word[100];
    int cost;
    int dp[100][100];
} WordCost;

int compare(const void *a, const void *b) {
    return ((WordCost *)a)->cost - ((WordCost *)b)->cost;
}

void insertWordCost(WordCost wordCosts[], int *wordCount, WordCost newWordCost) {
    if (*wordCount < 5) {
        wordCosts[*wordCount] = newWordCost;
        (*wordCount)++;
        qsort(wordCosts, *wordCount, sizeof(WordCost), compare);
    } else if (newWordCost.cost < wordCosts[4].cost) {
        wordCosts[4] = newWordCost;
        qsort(wordCosts, 5, sizeof(WordCost), compare);
    }
}

// 插入字符到字符串中的指定位置
void insertChar(char *str, char ch, int pos) {
    int len = strlen(str);
    // 檢查插入位置是否合法
    if (pos < 0 || pos > len) {
        printf("插入位置無效！\n");
        return;
    }
    // 使用 memmove 將字串向右移動一位
    memmove(str + pos + 1, str + pos, len - pos + 1);
    // // 將插入位置之後的字符向右移動一位
    // for (int i = len; i >= pos; i--) {
    //     str[i + 1] = str[i];
    // }
    // 插入字元
    str[pos] = ch;
}

int findMin(int a, int b, int c){
    if(a <= b && a <= c) return 0;
    if(b <= a && b <= c) return 1;
    return 2;
}

// 根據建好的dp查尋最短路徑
void findPath(WordCost *wordCost){
    int j = strlen(wordCost->origin_word);
    for(int i = strlen(targetString); i > 0; i--){
        // printf("i: %d, j: %d dp:%d dpi-1 j-1:%d\n", i, j, wordCost->dp[i][j], wordCost->dp[i-1][j-1]);
        if(j == 0){
            insertChar(wordCost->new_word, '+', j);
            continue;
        } else {
            int tmp = findMin(wordCost->dp[i][j-1], wordCost->dp[i-1][j], wordCost->dp[i-1][j-1]);
            switch (tmp) {
                case 0: // 如果是刪除該字元就替換成 -
                    insertChar(wordCost->new_word, '-', j);
                    break;
                case 1: // 如果是插入該字元就替換成 +
                    insertChar(wordCost->new_word, '+', j);
                    j++;
                    break;
                case 2: // 如果是不相同的字元就換成 *
                    if(wordCost->dp[i][j] != wordCost->dp[i-1][j-1]){
                        wordCost->new_word[j-1] = '*';
                    }
                    break;
                default:
                    break;
            }
            j--;
        }
    // printf("New String: %s\n\n", wordCost->new_word);
    }
}

int main(){
    FILE *fp;
    fp = fopen("wordlist.txt", "r");
    if(fp == NULL){
        printf("File not found\n");
        return 0;
    }
    char word[100];
    WordCost wordCosts[5]; // 只存儲前五個最小編輯距離的單詞
    for(int k = 0; k < 5; k++) {
        for(int i = 0; i < 100; i++) {
            for(int j = 0; j < 100; j++) {
                wordCosts[k].dp[i][j] = -1;
            }
        }
    }
    int wordCount = 0;
    int n = strlen(targetString);
    // 開始計算時間
    clock_t start, end;
    start = clock();

    // 一行一行讀取檔案
    while(fgets(word, 100, fp) != NULL){
        // 去除換行符號
        word[strlen(word)-1] = '\0';
        // 計算兩個字串的長度
        int m = strlen(word);
        // 存儲結果
        WordCost newWordCost;
        for (int i = 0; i < 100; i++){
            for (int j = 0; j < 100; j++){
                newWordCost.dp[i][j] = -1;
            }
        }
        strcpy(newWordCost.origin_word, word);
        strcpy(newWordCost.new_word, word);
        // 初始化陣列
        for(int i = 0; i <= n; i++) {
            for(int j = 0; j <= m; j++) {
                if(i == 0) {
                    newWordCost.dp[i][j] = j; // 將 targetString 的前 0 個字符轉換為 word 的前 j 個字符所需的編輯操作數
                } else if(j == 0) {
                    newWordCost.dp[i][j] = i; // 將 targetString 的前 i 個字符轉換為 word 的前 0 個字符所需的編輯操作數
                } else if(targetString[i-1] == word[j-1]) {
                    newWordCost.dp[i][j] = newWordCost.dp[i-1][j-1]; // 如果字符相同，則不需要編輯操作
                } else {
                    newWordCost.dp[i][j] = 1 + min(newWordCost.dp[i-1][j],    // 刪除 cost 1
                                                    newWordCost.dp[i][j-1],    // 插入 cost 1
                                                    newWordCost.dp[i-1][j-1] +1); // 替換 cost 2
                }
            }
        }
        newWordCost.cost = newWordCost.dp[n][m];
        insertWordCost(wordCosts, &wordCount, newWordCost);
    }
    // 打印前五個編輯距離最小的句子
    printf("Top 5 sentences with minimum edit distance:\n");
    // // 印出全部dp
    // for (int i = n; i >= 0; i--)
    // {
    //    for(int j = strlen(wordCosts[0].origin_word); j >=0 ; j--){
    //        printf("%02d ", wordCosts[0].dp[i][j]);
    //    }
    //    printf("\n");
    // }
    
    for(int i = 0; i < 5 && i < wordCount; i++) {
        // printf("%s\n",wordCosts[i].new_word);
        findPath(&wordCosts[i]);
        printf("Candidate #%d: %s ## MED %d ## OptPath: %s\n",i, wordCosts[i].origin_word, wordCosts[i].cost,wordCosts[i].new_word);
    }
     // 結束計算時間
     end = clock();
     double time_taken = ((double) (end - start)) / CLOCKS_PER_SEC /1000;
     printf("Time taken: %f ms\n", time_taken);
    fclose(fp);
    return 0;
}