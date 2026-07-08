// seq_detector_1011 — MUTANT (bug07_overlap_wrong_state.v) — after a detect the FSM goes to S2 instead of S1 (false detects on 10111x) (Mealy, overlapping)
module seq_detector_1011 (
    input  wire clk,
    input  wire rst_n,
    input  wire din,
    output wire detected
);
    localparam S0 = 2'd0, S1 = 2'd1, S2 = 2'd2, S3 = 2'd3;
    reg [1:0] state, next;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= S0;
        else        state <= next;
    end

    always @(*) begin
        case (state)
            S0: next = din ? S1 : S0;
            S1: next = din ? S1 : S2;
            S2: next = din ? S3 : S0;
            S3: next = din ? S2 : S2;  // BUG: wrong overlap state
            default: next = S0;
        endcase
    end

    assign detected = (state == S3) && din;
endmodule
