// counter_8bit — MUTANT (bug04_overflow_off_by_one.v) — up-count overflow pulses at 0xFE->0xFF instead of 0xFF->0x00
module counter_8bit (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       load,
    input  wire [7:0] d,
    input  wire       en,
    input  wire       up_down,
    output reg  [7:0] q,
    output reg        overflow
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            q        <= 8'h00;
            overflow <= 1'b0;
        end else if (load) begin
            q        <= d;
            overflow <= 1'b0;
        end else if (en) begin
            if (up_down) begin
                q        <= q + 8'h01;
                overflow <= (q == 8'hFE);  // BUG: one early
            end else begin
                q        <= q - 8'h01;
                overflow <= (q == 8'h00);
            end
        end else begin
            overflow <= 1'b0;
        end
    end
endmodule
